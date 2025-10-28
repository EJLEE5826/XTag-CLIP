import json
import logging
import math
import os
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip_train.distributed import is_master
from others.zero_shot_other import zero_shot_eval
from open_clip_train.precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, tag_loss, ce_loss, epoch, optimizer, scaler, scheduler, dist_model, args, tokenizer, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['scar_train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['scar_train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_additional, accum_tag_words, accum_tag_logit, accum_features = [], [], [], [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts, additional, text_prompt_tokens, class_words = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        additional = additional.to(device=device, non_blocking=True)
        if args.prompt_template_setting == "sentence_1":
            text_prompt_token = text_prompt_tokens[0].to(device=device, non_blocking=True)
        elif args.prompt_template_setting == "sentence_2":
            text_prompt_token = text_prompt_tokens[1].to(device=device, non_blocking=True)
        elif args.prompt_template_setting == "sentence_3":
            text_prompt_token = text_prompt_tokens[2].to(device=device, non_blocking=True)
        elif args.prompt_template_setting == "itemization":
            text_prompt_token = text_prompt_tokens[3].to(device=device, non_blocking=True)
        elif args.prompt_template_setting == "sentence_4":
            text_prompt_token = text_prompt_tokens[4].to(device=device, non_blocking=True)
        elif args.prompt_template_setting == "total":
            text_prompt_token = random.choice(text_prompt_tokens).to(device=device, non_blocking=True)
        else:
            text_prompt_token = None

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, text_prompt_token, additional, tokenizer, class_words)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                out_tagging_words = model_out.pop("tagging_words")
                out_tag_logits = model_out.pop("tag_logits")
                i2t_cls = model_out.pop("i2t_cls", None)
                t2i_cls = model_out.pop("t2i_cls", None)
                losses = loss(**model_out, output_dict=True)
                target_tag = additional.repeat(1,2)
                tag_losses = tag_loss(out_tag_logits, target_tag)
                ce_loss1 = ce_loss(i2t_cls) if i2t_cls is not None else torch.tensor(0.0)
                ce_loss2 = ce_loss(t2i_cls) if t2i_cls is not None else torch.tensor(0.0)
                ce_losses = ce_loss1 + ce_loss2

                losses["tagging_loss"] = tag_losses
                losses["ce_loss"] = ce_losses
                total_loss = sum(losses.values()) + tag_losses + ce_losses
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, text_prompt_token, additional, tokenizer, class_words)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    accum_tag_logit.append(model_out["tag_logits"])
                    model_out.pop("tag_logits", None)
                    accum_tag_words.append(model_out["tagging_words"])
                    model_out.pop("tagging_words", None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)
                accum_additional.append(additional)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                additional = accum_additional[j]
                with autocast():
                    model_out = model(images, text_prompt_token, additional, tokenizer, class_words)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    tag_losses = tag_loss(model_out["tag_logits"], additional)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values()) + tag_losses
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            if batch_size != args.batch_size:
                num_samples = dataloader.num_samples - batch_size
            else:
                num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{batch_count:>{sample_digits}}/{num_batches_per_epoch} ({percent_complete:.0f}%)] "
                f"Sample index : {num_samples}/{samples_per_epoch} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, loss, tag_loss, ce_loss, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    tagging_gt = []
    tagging_prid = []
    tagging_list = [
                    "Linear Width",
                    "Widened Width",
                    "Linear bulging Width",
                    "Normal Color",
                    "Pink Color",
                    "Red Color",
                    "Purple Color",
                    "Normal Pigmentation",
                    "Pigmented Pigmentation",
                    "Hypopigmented Pigmentation",
                    "Flat Surface",
                    "Hypertrophic Surface",
                    "Keloid Surface",
                    "Atrophic Surface",
                    "no Irregular Color",
                    "mild Irregular Color",
                    "moderate Irregular Color",
                    "severe Irregular Color",
                    "no Irregular Height",
                    "mild Irregular Height",
                    "moderate Irregular Height",
                    "severe Irregular Height"
                ]
    tag_metrics_sum = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "total_samples" : 0,
        "groups": {
            "Width": {"accuracy":0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "Color": {"accuracy":0.0,"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "Pigmentation": {"accuracy":0.0,"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "Surface": {"accuracy":0.0,"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "Irregular Color": {"accuracy":0.0,"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "Irregular Height": {"accuracy":0.0,"precision": 0.0, "recall": 0.0, "f1": 0.0}
        }
    }
    tag_metrics_avg = None
    
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    
    model.eval()

    val_output, classifier = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'scar_train' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['scar_train'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        
        
        with torch.inference_mode():
            top1, top2, n, finial_loss = 0., 0., 0., 0.
            per_class_acc = {1: torch.tensor([0]).to(device), 2: torch.tensor([0]).to(device)}
            per_class_counts = torch.tensor([0]).to(device)
            for i, batch in enumerate(dataloader):
                images, texts, additional, text_prompt_tokens, class_words = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)
                additional = additional.to(device=device, non_blocking=True)
                if args.prompt_template_setting == "sentence_1":
                    text_prompt_token = text_prompt_tokens[0].to(device=device, non_blocking=True)
                elif args.prompt_template_setting == "sentence_2":
                    text_prompt_token = text_prompt_tokens[1].to(device=device, non_blocking=True)
                elif args.prompt_template_setting == "sentence_3":
                    text_prompt_token = text_prompt_tokens[2].to(device=device, non_blocking=True)
                elif args.prompt_template_setting == "sentence_4":
                    text_prompt_token = text_prompt_tokens[3].to(device=device, non_blocking=True)
                elif args.prompt_template_setting == "itemization":
                    text_prompt_token = text_prompt_tokens[4].to(device=device, non_blocking=True)
                elif args.prompt_template_setting == "total":
                    text_prompt_token = random.choice(text_prompt_tokens).to(device=device, non_blocking=True)
                else:
                    raise ValueError("Invalid prompt template setting.")
                
                batch_size = images.size(0)
                with autocast():
                    model_out = model(images, text_prompt_token, additional, tokenizer, class_words)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    tagging_words = model_out.pop("tagging_words")
                    tag_logits = model_out.pop("tag_logits")
                    i2t_cls = model_out.pop("i2t_cls", None)
                    t2i_cls = model_out.pop("t2i_cls", None)
                    logits = 100. * image_features @ classifier
                    
                    tagging_gt.append(additional)
                    tagging_prid.extend(tagging_words)
                    
                    losses = loss(**model_out, output_dict=True)
                    target_tag = additional.repeat(1,2)
                    tag_losses = tag_loss(tag_logits, target_tag)
                    ce_loss1 = ce_loss(i2t_cls) if i2t_cls is not None else 0.0
                    ce_loss2 = ce_loss(t2i_cls) if t2i_cls is not None else 0.0
                    ce_losses = ce_loss1 + ce_loss2
                    
                    losses["tagging_loss"] = tag_losses
                    losses["ce_loss"] = ce_losses
                    total_loss = sum(losses.values()) + tag_losses + ce_losses
                    finial_loss += total_loss.item()
                    losses["loss"] = total_loss
                acc, class_counts, class_accuracy = accuracy(logits, texts, topk=(1, 2), 
                                  onehot_target=True if len(texts[0]) > 1 else False)
                
                tag_acc = calculate_batch_metrics(additional, tagging_words, tagging_list)
            
                tag_metrics_sum["accuracy"] += tag_acc["accuracy"] * batch_size
                tag_metrics_sum["precision"] += tag_acc["precision"] * batch_size
                tag_metrics_sum["recall"] += tag_acc["recall"] * batch_size
                tag_metrics_sum["f1"] += tag_acc["f1"] * batch_size
                tag_metrics_sum["total_samples"] += batch_size
                # 그룹별 메트릭 누적
                for group_name, group_metrics in tag_acc["groups"].items():
                    if group_name in tag_metrics_sum["groups"]:
                        tag_metrics_sum["groups"][group_name]["accuracy"] += group_metrics["accuracy"] * batch_size
                        tag_metrics_sum["groups"][group_name]["precision"] += group_metrics["precision"] * batch_size
                        tag_metrics_sum["groups"][group_name]["recall"] += group_metrics["recall"] * batch_size
                        tag_metrics_sum["groups"][group_name]["f1"] += group_metrics["f1"] * batch_size
                acc1, acc2 = acc
                top1 += acc1
                top2 += acc2
                n += images.size(0)

                per_class_acc[1] = per_class_acc[1] + class_accuracy[1]
                per_class_acc[2] = per_class_acc[2] + class_accuracy[2]
                per_class_counts = per_class_counts + class_counts   
                    
            tagging_gt = get_selected_items(torch.cat(tagging_gt,dim=0), tagging_list)
            tagging_output = [f"{a} - {b}" for a, b in zip(tagging_gt, tagging_prid)]
            
            logging.info(f"train data val class_counts: {per_class_counts.cpu().numpy().tolist()}", )
            logging.info(f"train data val tagging_output: {tagging_output[0]}")
            
            if tag_metrics_sum["total_samples"] > 0:
                tag_metrics_avg = {
                    "accuracy": tag_metrics_sum["accuracy"] / tag_metrics_sum["total_samples"],
                    "precision": tag_metrics_sum["precision"] / tag_metrics_sum["total_samples"],
                    "recall": tag_metrics_sum["recall"] / tag_metrics_sum["total_samples"],
                    "f1": tag_metrics_sum["f1"] / tag_metrics_sum["total_samples"],
                    "groups": {}
                }
                
                for group_name, group_metrics in tag_metrics_sum["groups"].items():
                    tag_metrics_avg["groups"][group_name] = {
                        "accuracy": group_metrics["accuracy"] / tag_metrics_sum["total_samples"],
                        "precision": group_metrics["precision"] / tag_metrics_sum["total_samples"],
                        "recall": group_metrics["recall"] / tag_metrics_sum["total_samples"],
                        "f1": group_metrics["f1"] / tag_metrics_sum["total_samples"]
                    }
                
                # 태깅 메트릭 로깅
                logging.info(f"train data val top1 accuracy: {top1 / n:.4f}")
                logging.info(f"train data val top2 accuracy: {top2 / n:.4f}")
                logging.info(f"train data val Tag accuracy: {tag_metrics_avg['accuracy']:.4f}")
                logging.info(f"train data val Tag F1 score: {tag_metrics_avg['f1']:.4f}")
                
                for group_name, group_metrics in tag_metrics_avg["groups"].items():
                    logging.info(f"{group_name} group - accuracy: {group_metrics['accuracy']:.4f}, "
                                f"F1: {group_metrics['f1']:.4f}," 
                                f"Precision: {group_metrics['precision']:.4f}, "
                                f"Recall: {group_metrics['recall']:.4f}")    
                    
        tagging_file = os.path.join(args.logs, args.name, "traindata_val_tagging_output.txt")
        with open(tagging_file, "a+") as f:
            for item in tagging_output:
                f.write("%s\n" % item)
            if tag_metrics_avg is not None:
                f.write(f"전체 정확도: {tag_metrics_avg['accuracy']:.4f} - ")
                f.write(f"전체 정밀도: {tag_metrics_avg['precision']:.4f} - ")
                f.write(f"전체 재현율: {tag_metrics_avg['recall']:.4f} - ")
                f.write(f"전체 F1 점수: {tag_metrics_avg['f1']:.4f}\n")
                
                f.write("그룹별 메트릭:\n")
                for group_name, group_metrics in tag_metrics_avg['groups'].items():
                    f.write(f"      {group_name} 그룹: ")
                    f.write(f"정확도: {group_metrics['accuracy']:.4f} - ")
                    f.write(f"F1 점수: {group_metrics['f1']:.4f} - ")
                    f.write(f"정밀도: {group_metrics['precision']:.4f} - ")
                    f.write(f"재현율: {group_metrics['recall']:.4f}\n")
                f.write("\n")
        for k in (1, 2):
            #print(per_class_acc[k].device, class_counts.device)
            per_class_acc[k] = per_class_acc[k] / per_class_counts.clamp(min=1)
            formatted_values = [f"{val:.4f}" for val in per_class_acc[k].cpu().numpy()]
            logging.info(f'per_class_acc(top-{k}): {", ".join(formatted_values)}')
    else:
        top1, finial_loss, n = 0., 10e5, 10
        tag_metrics_avg = {
            "accuracy": 0.0,
            }
    
    return val_output, top1 / n, finial_loss / n, tag_metrics_avg['accuracy']
    
def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
    
def get_selected_items(tensor_list, reference_list):
    """
    텐서 요소를 가진 리스트에서 선택된 항목을 추출하는 함수
    
    Args:
        tensor_list: 텐서로 구성된 리스트, 각 텐서는 이진값(0,1)을 가진 [num_classes] 형태
        reference_list: 참조할 문자열 리스트 (길이는 각 텐서의 길이와 일치해야 함)
        
    Returns:
        각 텐서의 1인 위치에 해당하는 참조 항목을 쉼표로 구분한 문자열 리스트
    """
    results = []
    
    # 각 텐서에 대해 처리
    for tensor in tensor_list:
        # 텐서의 1인 위치 인덱스 추출
        indices = torch.nonzero(tensor == 1, as_tuple=True)[0]
        indices = indices.cpu().numpy().tolist()
        
        # 선택된 항목 추출
        selected_items = [reference_list[idx] for idx in indices]
        results.append(",".join(selected_items))
    
    return results

def calculate_batch_metrics(true_binary_tensor, predicted_items_strings, reference_list, group_sizes=[3, 4, 3, 4, 4, 4]):
    """
    배치 데이터에 대한 그룹별 정확도 및 메트릭을 계산합니다.
    
    Args:
        true_binary_tensor: 정답 이진값 텐서 [batch_size, class_num]
        predicted_items_strings: 예측 항목 문자열 리스트 [batch_size]의 리스트, 각 항목은 "a,c"와 같은 형태
        reference_list: 전체 참조 리스트 [class_num] (ex: ["a","b","c","d","e","f"])
        group_sizes: 각 그룹별 클래스 개수 리스트 (ex: [3, 4, 3, 3, 3, 6])
        
    Returns:
        전체 및 그룹별 메트릭 딕셔너리
    """
    class_num = true_binary_tensor.shape[1]
    
    # 입력 검증
    if sum(group_sizes) != class_num:
        logging.warning(f"그룹 크기의 합({sum(group_sizes)})이 클래스 수({class_num})와 일치하지 않습니다. 기본값으로 진행합니다.")
    
    # 예측 항목 문자열을 이진 텐서로 변환
    predicted_binary_tensor = torch.zeros_like(true_binary_tensor)
    
    for i, items_str in enumerate(predicted_items_strings):
        if items_str:  # 빈 문자열이 아닌 경우
            items = items_str.split(',')
            for item in items:
                if item in reference_list:
                    idx = reference_list.index(item)
                    predicted_binary_tensor[i, idx] = 1
    
    
    # 혼동 행렬 요소 계산 (텐서 연산 활용)
    tp = ((true_binary_tensor == 1) & (predicted_binary_tensor == 1)).float().sum(dim=1)
    tn = ((true_binary_tensor == 0) & (predicted_binary_tensor == 0)).float().sum(dim=1)
    fp = ((true_binary_tensor == 0) & (predicted_binary_tensor == 1)).float().sum(dim=1)
    fn = ((true_binary_tensor == 1) & (predicted_binary_tensor == 0)).float().sum(dim=1)
    
    # 1(양성)에 초점을 맞춘 정확도 계산 - TP를 모든 양성 예측과 실제 양성 중 비율로 계산
    # accuracy_1focused는 (TP) / (TP + FP + FN) 형태로 1인 경우에 대한 정확도만 고려
    positive_accuracy = tp / (tp + fp + fn + 1e-8)
    
    # 샘플별 메트릭 계산
    precision = tp / (tp + fp + 1e-8)  # 0으로 나누기 방지
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # 배치 평균 메트릭
    avg_metrics = {
        "accuracy": positive_accuracy.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item()
    }
    
    # 그룹 이름 정의
    group_names = [
        "Width", "Color", "Pigmentation", "Surface", "Irregular Color", "Irregular Height"
    ][:len(group_sizes)]
    
    # 그룹별 메트릭 계산
    group_metrics = {}
    start_idx = 0
    
    for g_idx, group_size in enumerate(group_sizes):
        if start_idx + group_size > class_num:
            break
            
        end_idx = start_idx + group_size
        group_name = group_names[g_idx] if g_idx < len(group_names) else f"Group {g_idx+1}"
        
        # 그룹 내 클래스에 대한 메트릭 계산
        group_tp = ((true_binary_tensor[:, start_idx:end_idx] == 1) & 
                   (predicted_binary_tensor[:, start_idx:end_idx] == 1)).float().sum(dim=1)
        group_fp = ((true_binary_tensor[:, start_idx:end_idx] == 0) & 
                   (predicted_binary_tensor[:, start_idx:end_idx] == 1)).float().sum(dim=1)
        group_fn = ((true_binary_tensor[:, start_idx:end_idx] == 1) & 
                   (predicted_binary_tensor[:, start_idx:end_idx] == 0)).float().sum(dim=1)
        
        group_positive_accuracy = group_tp / (group_tp + group_fp + group_fn + 1e-8)
        
        # 그룹별 정밀도, 재현율, F1 계산
        group_precision = group_tp / (group_tp + group_fp + 1e-8)
        group_recall = group_tp / (group_tp + group_fn + 1e-8)
        group_f1 = 2 * (group_precision * group_recall) / (group_precision + group_recall + 1e-8)
        
        # 그룹 메트릭 저장
        group_metrics[group_name] = {
            "accuracy": group_positive_accuracy.mean().item(),
            "precision": group_precision.mean().item(),
            "recall": group_recall.mean().item(),
            "f1": group_f1.mean().item()
        }
        
        # 다음 그룹의 시작 인덱스로 업데이트
        start_idx = end_idx
    
    # 전체 메트릭과 그룹 메트릭을 결합
    avg_metrics["groups"] = group_metrics
    
    return avg_metrics

def accuracy(output, target, topk=(1,), onehot_target=False):

    if not onehot_target:
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

    else:
        batch_size, num_classes = output.size()

        # top-k ���� ��� (batch_size x maxk)
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        
        # target�� �ݵ�� boolean ���°� �ƴϴ���, ���� 1�� ���� True�� �Ǵ�
        target_bool = target.bool()
        
        # pred�� �� ��ġ�� target���� True���� Ȯ�� (batch_size x maxk)
        correct = target_bool.gather(1, pred)

        # �� ���ø��� top-k ���� ��� �� �ϳ��� True�̸� �ش� ������ �������� ó��
        overall_acc = [correct[:, :k].any(dim=1).float().sum().item() for k in topk]


        class_counts = target_bool.sum(dim=0)
        class_accuracy = {}
        for k in topk:
            # pred[:, :k]: shape (batch_size, k)
            # �񱳸� ���� �� Ŭ���� �ε����� (1,1,num_classes)�� Ȯ��
            arange = torch.arange(num_classes, device=output.device).view(1, 1, num_classes)
            # top-k ���� ����� (batch_size, k, 1)�� Ȯ��
            pred_k = pred[:, :k].unsqueeze(2)
            # �� ���ÿ� ����, �� Ŭ������ top-k ������ ���ԵǾ����� �� (batch_size, k, num_classes)
            pred_eq = (pred_k == arange)
            # k ������ ���� any() ���� �� (batch_size, num_classes): �� ���ø��� �� Ŭ������ top-k ������ ���ԵǾ����� ����
            pred_in_topk = pred_eq.any(dim=1)
            # �� Ŭ��������, �ش� Ŭ������ ������ ���� �� top-k ������ ���Ե� sample ��
            correct_class = (target_bool & pred_in_topk).sum(dim=0)
            # �� Ŭ������ ����, positive sample �� �� %�� top-k ������ ���ԵǾ����� ��� (0���� ������ ��츦 �����ϱ� ���� clamp)
            class_accuracy[k] = correct_class.float()
            # acc = correct_class.float() / class_counts.float().clamp(min=1)
            # class_accuracy[k] = acc  # tensor shape: (num_classes,)
            
        return overall_acc, class_counts.float(), class_accuracy
