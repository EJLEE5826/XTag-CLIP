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

from sklearn import metrics as sk_metrics

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP, SCAR_CLASSNAMES, SIMPLE_SCAR_TEMPLATES, get_tokenizer, build_zero_shot_classifier
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
                text_features_l = model_out.pop("text_features_l", None)
                text_features_g = model_out.pop("text_features_g", None)
                image_features_l = model_out.pop("image_features_l", None)
                image_features_g = model_out.pop("image_features_g", None)
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
    metrics_vla = {}
    tagging_gt = []
    tagging_gt_val = []
    class_gt_val = []
    class_prid_val = []
    class_prid_score_val = []
    tagging_prid = []
    tagging_prid_val = []
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
    tag_metrics_sum_val = tag_metrics_sum.copy()
    tag_metrics_avg = None
    tag_metrics_avg_val = None
    
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    
    model.eval()

    #val_output, classifier = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)

    if args.distributed and not args.horovod:
        model = model.module

    #assert (len(data.keys()))==1

    if "scar_val" in data:
        data_name = "scar_val"
        classnames = SCAR_CLASSNAMES
        templates = SIMPLE_SCAR_TEMPLATES
    else:
        raise ValueError("Invalid dataset name")
    
    # logging.info(f'Starting zero-shot {','.join(list(data.keys()))}.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)
            
    if args.zeroshot_frequency == 0:
        return 0. , None
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return 0., None
    
    logging.info('Building classifier')
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=classnames,
            templates=templates,
            num_classes_per_batch=3,
            device=device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    
    logging.info(f'Starting validation {data_name}.')
    if 'scar_val' in data:
        dataloader_val = data['scar_val'].dataloader
        num_samples_val = 0
        samples_per_val = dataloader_val.num_samples
        per_class_counts_val = torch.zeros(len(SCAR_CLASSNAMES)).to(device)
        
        with torch.inference_mode():
            top1_val, top2_val, n_val, finial_loss_val = 0., 0., 0., 0.
            per_class_correct_val = {1: torch.zeros(len(SCAR_CLASSNAMES)).to(device), 
                    2: torch.zeros(len(SCAR_CLASSNAMES)).to(device)}
            per_class_total_val = {1: torch.zeros(len(SCAR_CLASSNAMES)).to(device), 
                    2: torch.zeros(len(SCAR_CLASSNAMES)).to(device)}
            for i, batch_val in enumerate(dataloader_val):
                images_val, texts_val, additional_val, text_prompt_tokens_val, class_words_val = batch_val
                images_val = images_val.to(device=device, dtype=get_input_dtype(args.precision), non_blocking=True)
                texts_val = texts_val.to(device=device, non_blocking=True)
                additiona_val = additional_val.to(device=device, non_blocking=True)
                if args.prompt_template_setting == "sentence_1":
                    text_prompt_token_val = text_prompt_tokens_val[0].to(device=device, non_blocking=True)
                elif args.prompt_template_setting == "sentence_2":
                    text_prompt_token_val = text_prompt_tokens_val[1].to(device=device, non_blocking=True)
                elif args.prompt_template_setting == "sentence_3":
                    text_prompt_token_val = text_prompt_tokens_val[2].to(device=device, non_blocking=True)
                elif args.prompt_template_setting == "sentence_4":
                    text_prompt_token_val = text_prompt_tokens_val[3].to(device=device, non_blocking=True)
                elif args.prompt_template_setting == "itemization":
                    text_prompt_token_val = text_prompt_tokens_val[4].to(device=device, non_blocking=True)
                elif args.prompt_template_setting == "total":
                    text_prompt_token_val = random.choice(text_prompt_tokens_val).to(device=device, non_blocking=True)
                else:
                    raise ValueError("Invalid prompt template setting.")
                
                batch_size_val = images_val.size(0)
                with autocast():
                    model_out_val = model(images_val, text_prompt_token_val, additional_val, tokenizer, class_words_val)
                    image_features_val = model_out_val["image_features"]
                    text_features_val = model_out_val["text_features"]
                    logit_scale_val = model_out_val["logit_scale"]
                    tagging_words_val = model_out_val.pop("tagging_words")
                    tag_logits_val = model_out_val.pop("tag_logits")
                    i2t_cls_val = model_out_val.pop("i2t_cls", None)
                    t2i_cls_val = model_out_val.pop("t2i_cls", None)
                    text_features_l_val = model_out_val.pop("text_features_l", None)
                    text_features_g_val = model_out_val.pop("text_features_g", None)
                    image_features_l_val = model_out_val.pop("image_features_l", None)
                    image_features_g_val = model_out_val.pop("image_features_g", None)
                    if args.use_fusion:
                        image_features_l_val = image_features_l_val.detach().cpu().numpy()
                        image_features_g_val = image_features_g_val.detach().cpu().numpy()
                        text_features_l_val = text_features_l_val.detach().cpu().numpy()
                        text_features_g_val = text_features_g_val.detach().cpu().numpy()
                        global_similarity_val = sk_metrics.pairwise.cosine_similarity(image_features_g_val, classifier.T.detach().cpu().numpy())
                        local_similarity_val = []

                        for z in range(image_features_l_val.shape[1]):
                            sim_val = sk_metrics.pairwise.cosine_similarity(image_features_l_val[:, z, :], classifier.T.detach().cpu().numpy())
                            local_similarity_val.append(sim_val)
                        local_similarity_val = np.stack(local_similarity_val, axis=0).mean(axis=0)
                        similarity_val = 100. * (global_similarity_val + local_similarity_val) / 2
                        similarity_val = torch.tensor(similarity_val, device=device, dtype=torch.float32)

                    else:
                        similarity_val = 100. * image_features_val @ classifier
                    
                    tagging_gt_val.append(additional_val)
                    class_gt_val.extend(texts_val)
                    tagging_prid_val.extend(tagging_words_val)
                    batch_max_to_one_liner = lambda tensor, dim=-1: (tensor == torch.max(tensor, dim=dim, keepdim=True)[0]).float()
                    class_prid_score_val.extend(similarity_val)
                    class_prid_val.extend(batch_max_to_one_liner(similarity_val, dim=1))

                    losses_val = loss(**model_out_val, output_dict=True)
                    target_tag_val = additional_val.repeat(1,2)
                    tag_losses_val = tag_loss(tag_logits_val, target_tag_val)
                    ce_loss1_val = ce_loss(i2t_cls_val) if i2t_cls_val is not None else 0.0
                    ce_loss2_val = ce_loss(t2i_cls_val) if t2i_cls_val is not None else 0.0
                    ce_losses_val = ce_loss1_val + ce_loss2_val
                    losses_val["tagging_loss"] = tag_losses_val
                    losses_val["ce_loss"] = ce_losses_val
                    total_loss_val = sum(losses_val.values()) + tag_losses_val + ce_losses_val
                    finial_loss_val += total_loss_val.item()
                acc_val, class_counts_val, class_metrics_val, f1_metrics_val = accuracy(similarity_val, texts_val, topk=(1, 2),
                                    onehot_target=True if len(texts_val[0]) > 1 else False)
                tag_acc_val = calculate_batch_metrics(additional_val, tagging_words_val, tagging_list)
                tag_metrics_sum_val["accuracy"] += tag_acc_val["accuracy"] * batch_size_val
                tag_metrics_sum_val["precision"] += tag_acc_val["precision"] * batch_size_val
                tag_metrics_sum_val["recall"] += tag_acc_val["recall"] * batch_size_val
                tag_metrics_sum_val["f1"] += tag_acc_val["f1"] * batch_size_val
                tag_metrics_sum_val["total_samples"] += batch_size_val
                # 그룹별 메트릭 누적
                for group_name, group_metrics in tag_acc_val["groups"].items():
                    if group_name in tag_metrics_sum_val["groups"]:
                        tag_metrics_sum_val["groups"][group_name]["accuracy"] += group_metrics["accuracy"] * batch_size_val
                        tag_metrics_sum_val["groups"][group_name]["precision"] += group_metrics["precision"] * batch_size_val
                        tag_metrics_sum_val["groups"][group_name]["recall"] += group_metrics["recall"] * batch_size_val
                        tag_metrics_sum_val["groups"][group_name]["f1"] += group_metrics["f1"] * batch_size_val
                acc1_val, acc2_val = acc_val
                top1_val += acc1_val
                top2_val += acc2_val
                n_val += images_val.size(0)
                for k in (1, 2):
                    per_class_correct_val[k] += class_metrics_val[k]['correct']
                    per_class_total_val[k] += class_metrics_val[k]['total']
                per_class_counts_val += class_counts_val
                
            tagging_gt_val = get_selected_items(torch.cat(tagging_gt_val,dim=0), tagging_list)
            tagging_output_val = [f"{a} - {b}" for a, b in zip(tagging_gt_val, tagging_prid_val)]
            
            class_gt_val = get_selected_items(class_gt_val, classnames)
            class_prid_val = get_selected_items(class_prid_val, classnames)
            class_output_val = [f"{a} - {b} - {c.tolist()}" for a, b, c in zip(class_gt_val, class_prid_val, class_prid_score_val)]
            
            logging.info(f"val data val class_counts: {per_class_counts_val.cpu().numpy().tolist()}", )
            logging.info(f"val data val tagging_output: {tagging_output_val[0]}")
            if tag_metrics_sum_val["total_samples"] > 0:
                tag_metrics_avg_val = {
                    "accuracy": tag_metrics_sum_val["accuracy"] / tag_metrics_sum_val["total_samples"],
                    "precision": tag_metrics_sum_val["precision"] / tag_metrics_sum_val["total_samples"],
                    "recall": tag_metrics_sum_val["recall"] / tag_metrics_sum_val["total_samples"],
                    "f1": tag_metrics_sum_val["f1"] / tag_metrics_sum_val["total_samples"],
                    "groups": {}
                }
                
                for group_name, group_metrics in tag_metrics_sum_val["groups"].items():
                    tag_metrics_avg_val["groups"][group_name] = {
                        "accuracy": group_metrics["accuracy"] / tag_metrics_sum_val["total_samples"],
                        "precision": group_metrics["precision"] / tag_metrics_sum_val["total_samples"],
                        "recall": group_metrics["recall"] / tag_metrics_sum_val["total_samples"],
                        "f1": group_metrics["f1"] / tag_metrics_sum_val["total_samples"]
                    }
                
                # 태깅 메트릭 로깅
                logging.info(f"val data val top1 accuracy: {top1_val / n_val:.4f}")
                logging.info(f"val data val top2 accuracy: {top2_val / n_val:.4f}")
                if f1_metrics_val is not None:
                    class_precision, class_recall, class_f1, overall_f1 = f1_metrics_val
                    logging.info(f'Overall F1 score (top-1): {overall_f1[1]:.4f}')
                    logging.info(f'Overall F1 score (top-2): {overall_f1[2]:.4f}')
                    logging.info(f"Class F1 scores (top-1): {', '.join([f'{val:.4f}' for val in class_f1[1].cpu().numpy()])}")
                logging.info(f"val data val Tag accuracy: {tag_metrics_avg_val['accuracy']:.4f}")
                logging.info(f"val data val Tag F1 score: {tag_metrics_avg_val['f1']:.4f}")
                
                for group_name, group_metrics in tag_metrics_avg_val["groups"].items():
                    logging.info(f"{group_name} group - accuracy: {group_metrics['accuracy']:.4f}, "
                                f"F1: {group_metrics['f1']:.4f}," 
                                f"Precision: {group_metrics['precision']:.4f}, "
                                f"Recall: {group_metrics['recall']:.4f}")  
        
        tagging_file = os.path.join(args.logs, args.name, "traindata_val_tagging_output.txt")
        class_file = os.path.join(args.logs, args.name, "traindata_val_class_output.txt")
        with open(tagging_file, "a+") as f:
            for item in tagging_output_val:
                f.write("%s\n" % item)
            if tag_metrics_avg_val is not None:
                f.write(f"전체 정확도: {tag_metrics_avg_val['accuracy']:.4f} - ")
                f.write(f"전체 정밀도: {tag_metrics_avg_val['precision']:.4f} - ")
                f.write(f"전체 재현율: {tag_metrics_avg_val['recall']:.4f} - ")
                f.write(f"전체 F1 점수: {tag_metrics_avg_val['f1']:.4f}\n")
                
                f.write("그룹별 메트릭:\n")
                for group_name, group_metrics in tag_metrics_avg_val['groups'].items():
                    f.write(f"      {group_name} 그룹: ")
                    f.write(f"정확도: {group_metrics['accuracy']:.4f} - ")
                    f.write(f"F1 점수: {group_metrics['f1']:.4f} - ")
                    f.write(f"정밀도: {group_metrics['precision']:.4f} - ")
                    f.write(f"재현율: {group_metrics['recall']:.4f}\n")
                f.write("\n")
        with open(class_file, "a+") as f:
            for item in class_output_val:
                f.write("%s\n" % item)
            f.write(f"val data val top1 accuracy: {top1_val / n_val:.4f}\n")
        
        for k in (1, 2):
            per_class_acc_val = per_class_correct_val[k] / per_class_total_val[k].clamp(min=1)
            formatted_values_val = [f"{val:.4f}" for val in per_class_acc_val.cpu().numpy()]
            logging.info(f'per_class_acc(top-{k}): {", ".join(formatted_values_val)}')
        
    else:
        top1_val = 0
        top2_val = 0
        n_val = 1
        finial_loss_val = 0.0
        tag_metrics_avg_val = None
        tagging_output_val = []
        tagging_gt_val = []
        tagging_prid_val = []
                
    
    input_dtype = get_input_dtype(args.precision)

    if 'scar_train' in data and (args.val_frequency and ((epoch % 10) == 0 or epoch == args.epochs)):
        dataloader = data['scar_train'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        
        per_class_counts = torch.zeros(len(SCAR_CLASSNAMES)).to(device)
        with torch.inference_mode():
            top1, top2, n, finial_loss = 0., 0., 0., 0.
            per_class_correct = {1: torch.zeros(len(SCAR_CLASSNAMES)).to(device), 
                    2: torch.zeros(len(SCAR_CLASSNAMES)).to(device)}
            per_class_total = {1: torch.zeros(len(SCAR_CLASSNAMES)).to(device), 
                    2: torch.zeros(len(SCAR_CLASSNAMES)).to(device)}
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
                    text_features_l = model_out.pop("text_features_l", None)
                    text_features_g = model_out.pop("text_features_g", None)
                    image_features_l = model_out.pop("image_features_l", None)
                    image_features_g = model_out.pop("image_features_g", None)
                    if args.use_fusion:
                        image_features_l = image_features_l.detach().cpu().numpy()
                        image_features_g = image_features_g.detach().cpu().numpy()
                        text_features_l = text_features_l.detach().cpu().numpy()
                        text_features_g = text_features_g.detach().cpu().numpy()
                        global_similarity = sk_metrics.pairwise.cosine_similarity(image_features_g, classifier.T.detach().cpu().numpy())
                        local_similarity = []
                        for z in range(image_features_l.shape[1]):
                            sim_ = sk_metrics.pairwise.cosine_similarity(image_features_l[:, z, :], classifier.T.detach().cpu().numpy())
                            local_similarity.append(sim_)
                        local_similarity = np.stack(local_similarity, axis=0).mean(axis=0)
                        similarity = 100. * (global_similarity + local_similarity) / 2
                        similarity = torch.tensor(similarity, device=device, dtype=torch.float32)
                    else:
                        similarity = 100. * image_features @ classifier
                    
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
                acc, class_counts, class_metrics, f1_metrics = accuracy(similarity, texts, topk=(1, 2), 
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

                for k in (1, 2):
                    per_class_correct[k] += class_metrics[k]['correct']
                    per_class_total[k] += class_metrics[k]['total']  
                per_class_counts += class_counts
                    
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
                if f1_metrics is not None:
                    class_precision, class_recall, class_f1, overall_f1 = f1_metrics
                    logging.info(f'Overall F1 score (top-1): {overall_f1[1]:.4f}')
                    logging.info(f'Overall F1 score (top-2): {overall_f1[2]:.4f}')
                    logging.info(f"Class F1 scores (top-1): {', '.join([f'{val:.4f}' for val in class_f1[1].cpu().numpy()])}")
                logging.info(f"train data val Tag accuracy: {tag_metrics_avg['accuracy']:.4f}")
                logging.info(f"train data val Tag F1 score: {tag_metrics_avg['f1']:.4f}")
                
                for group_name, group_metrics in tag_metrics_avg["groups"].items():
                    logging.info(f"{group_name} group - accuracy: {group_metrics['accuracy']:.4f}, "
                                f"F1: {group_metrics['f1']:.4f}," 
                                f"Precision: {group_metrics['precision']:.4f}, "
                                f"Recall: {group_metrics['recall']:.4f}")    
                    
        for k in (1, 2):
            per_class_acc = per_class_correct[k] / per_class_total[k].clamp(min=1)
            formatted_values = [f"{val:.4f}" for val in per_class_acc.cpu().numpy()]
            logging.info(f'per_class_acc(top-{k}): {", ".join(formatted_values)}')
    else:
        top1, finial_loss, n = 0., 10e5, 10
        tag_metrics_avg = {
            "accuracy": 0.0,
            }
    
    return top1_val / n_val, top1 / n, finial_loss / n, tag_metrics_avg['accuracy']
    
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
    
    # 정확도 계산 수정: 일반적인 정확도 공식 사용 (TP + TN) / (TP + TN + FP + FN)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    # 샘플별 메트릭 계산
    precision = tp / (tp + fp + 1e-8)  # 0으로 나누기 방지
    recall = tp / (tp + fn + 1e-8)
    
    # F1 계산 시 NaN 방지
    denominator = precision + recall
    f1 = torch.where(denominator > 0, 
                    2 * (precision * recall) / denominator, 
                    torch.zeros_like(denominator))
    
    # 배치 평균 메트릭
    avg_metrics = {
        "accuracy": accuracy.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item()
    }
    
    # 그룹 이름 정의 - 필요에 따라 수정 가능하도록 변수화
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
        
        # 그룹 내 클래스에 대한 혼동 행렬 요소 계산
        group_tp = ((true_binary_tensor[:, start_idx:end_idx] == 1) & 
                   (predicted_binary_tensor[:, start_idx:end_idx] == 1)).float().sum(dim=1)
        group_tn = ((true_binary_tensor[:, start_idx:end_idx] == 0) & 
                   (predicted_binary_tensor[:, start_idx:end_idx] == 0)).float().sum(dim=1)
        group_fp = ((true_binary_tensor[:, start_idx:end_idx] == 0) & 
                   (predicted_binary_tensor[:, start_idx:end_idx] == 1)).float().sum(dim=1)
        group_fn = ((true_binary_tensor[:, start_idx:end_idx] == 1) & 
                   (predicted_binary_tensor[:, start_idx:end_idx] == 0)).float().sum(dim=1)
        
        # 정확도 계산 수정
        group_accuracy = (group_tp + group_tn) / (group_tp + group_tn + group_fp + group_fn + 1e-8)
        
        # 그룹별 정밀도, 재현율 계산
        group_precision = group_tp / (group_tp + group_fp + 1e-8)
        group_recall = group_tp / (group_tp + group_fn + 1e-8)
        
        # F1 계산 시 NaN 방지
        group_denominator = group_precision + group_recall
        group_f1 = torch.where(group_denominator > 0,
                              2 * (group_precision * group_recall) / group_denominator,
                              torch.zeros_like(group_denominator))
        
        # 그룹 메트릭 저장
        group_metrics[group_name] = {
            "accuracy": group_accuracy.mean().item(),
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
    """
    정확도와 F1 점수를 계산합니다.
    
    Args:
        output: 모델의 출력 logits [batch_size, num_classes]
        target: 정답 레이블 [batch_size] 또는 one-hot 형태 [batch_size, num_classes]
        topk: 계산할 top-k 정확도의 k 값들
        onehot_target: target이 one-hot 인코딩 형태인지 여부
        
    Returns:
        onehot_target이 False인 경우: (accuracy_vals, None, None, None)
        onehot_target이 True인 경우: (overall_acc, class_counts, class_metrics, (class_precision, class_recall, class_f1, overall_f1))
    """
    
    if not onehot_target:
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # 원래 정확도 계산
        accuracy_vals = [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
        # 일관성 있는 반환 값 제공
        return accuracy_vals, torch.tensor([0]).to(output.device), {k: torch.tensor([0]).to(output.device) for k in topk}, None
    
    else:
        batch_size, num_classes = output.size()
        
        # top-k 예측 인덱스 (batch_size x maxk)
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        
        # target을 boolean 형태로 변환
        target_bool = target.bool()
        
        # 정확도 계산 부분
        correct = target_bool.gather(1, pred)
        overall_acc = [correct[:, :k].any(dim=1).float().sum().item() for k in topk]
        
        # 클래스별 메트릭 계산
        class_counts = target_bool.sum(dim=0)
        class_metrics = {}  # 변경: class_accuracy -> class_metrics
        class_precision = {}
        class_recall = {}
        class_f1 = {}
        
        # F1 점수 계산에 필요한 변수
        for k in topk:
            # 메모리 효율적 방식으로 수정: 한 번에 모든 클래스를 비교하는 대신 반복문 사용
            pred_in_topk = torch.zeros((batch_size, num_classes), device=output.device, dtype=torch.bool)
            for i in range(batch_size):
                pred_in_topk[i, pred[i, :k]] = True
            
            # True Positives: 실제 양성이고 예측도 양성인 경우
            tp = (target_bool & pred_in_topk).sum(dim=0)
            
            # False Positives: 실제 음성이나 예측은 양성인 경우
            fp = ((~target_bool) & pred_in_topk).sum(dim=0)
            
            # False Negatives: 실제 양성이나 예측은 음성인 경우
            fn = (target_bool & (~pred_in_topk)).sum(dim=0)
            
            # True Negatives: 실제 음성이고 예측도 음성인 경우
            tn = ((~target_bool) & (~pred_in_topk)).sum(dim=0)
            
            # 여기를 수정: 바로 정확도를 계산하는 대신 TP+TN과 TP+TN+FP+FN을 별도로 저장
            # 이 값들은 evaluate 함수에서 클래스별 정확도를 계산하는 데 사용됨
            class_metrics[k] = {
                'correct': tp,  # 정확히 예측한 샘플 수 (TP+TN)
                'total': tp + tn + fp + fn  # 전체 샘플 수
            }
            
            # 정밀도(precision): TP / (TP + FP)
            precision = tp.float() / (tp + fp).float().clamp(min=1e-8)
            class_precision[k] = precision
            
            # 재현율(recall): TP / (TP + FN)
            recall = tp.float() / (tp + fn).float().clamp(min=1e-8)
            class_recall[k] = recall
            
            # F1 스코어: 2 * (precision * recall) / (precision + recall)
            denominator = precision + recall
            f1 = torch.where(denominator > 0, 
                             2 * (precision * recall) / denominator, 
                             torch.zeros_like(denominator))
            class_f1[k] = f1
        
        # 전체 F1 스코어 (매크로 평균)
        overall_f1 = {k: class_f1[k].mean().item() for k in topk}
            
        return overall_acc, class_counts.float(), class_metrics, (class_precision, class_recall, class_f1, overall_f1)
