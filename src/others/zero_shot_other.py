import logging
import os
import torch
from tqdm import tqdm
import random

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, \
    SIMPLE_MEDICALMNIST_TEMPLATES, MEDICALMNIST_CLASSNAMES, PATHMNIST_CLASSNAMES, SCAR_CLASSNAMES, SIMPLE_SCAR_TEMPLATES
from open_clip_train.precision import get_autocast


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



def run(model, classifier, dataloader, args, tokenizer, label_value=0):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    # to save embeddings
    img_embeddings = []
    # txt_embeddings = []
    labels = []
    dataset_labels = []
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

    with torch.inference_mode():
        top1, top2, n = 0., 0., 0.
        per_class_acc = {1: torch.tensor([0]).to(device), 2: torch.tensor([0]).to(device)}
        per_class_counts = torch.tensor([0]).to(device)

        for i, (images, target, tagging, text_prompt_tokens, class_words ) in tqdm(enumerate(dataloader), total=len(dataloader)):
            images = images.to(device=device, dtype=input_dtype)
            target = target.to(device)
            tagging = tagging.to(device)
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
                # predict
                output = model(image=images, text=text_prompt_token, additional_label = tagging, tokenizer=tokenizer, class_words=class_words)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

                tagging_gt.append(tagging)
                tagging_prid.extend(output['tagging_words'])
                
                if args.save_embed:
                    # save embeddings
                    img_embeddings.append(image_features.cpu())
                    labels.extend(target.cpu().numpy())
                    dataset_labels.extend([label_value] * image_features.size(0))

            # measure accuracy
            acc, class_counts, class_accuracy = accuracy(logits, target, topk=(1, 2), 
                                  onehot_target=True if len(target[0]) > 1 else False)
            
            tag_acc = calculate_batch_metrics(tagging, output["tagging_words"], tagging_list)
            
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

        logging.info(f"validation class_counts: {per_class_counts.cpu().numpy().tolist()}", )
        tagging_gt = get_selected_items(torch.cat(tagging_gt, dim=0), tagging_list)
        tagging_output = [f"{a} - {b}" for a, b in zip(tagging_gt, tagging_prid)]
        
        logging.info(f"validation tagging_output: {tagging_output[0]}")
        
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
            logging.info(f"val data val top1 accuracy: {top1 / n:.4f}")
            logging.info(f"val data val top2 accuracy: {top2 / n:.4f}")

            logging.info(f"val data val Tag accuracy: {tag_metrics_avg['accuracy']:.4f}")
            logging.info(f"val data val Tag F1 score: {tag_metrics_avg['f1']:.4f}")
            
            for group_name, group_metrics in tag_metrics_avg["groups"].items():
                logging.info(f"{group_name} group - accuracy: {group_metrics['accuracy']:.4f}, "
                            f"F1: {group_metrics['f1']:.4f}," 
                            f"Precision: {group_metrics['precision']:.4f}, "
                            f"Recall: {group_metrics['recall']:.4f}")    
                
    tagging_file = os.path.join(args.logs, args.name, "val_data_tagging_output.txt")
    with open(tagging_file, "a") as f:
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

    top1 = (top1 / n)
    top2 = (top2 / n)

    if args.save_embed:
        all_img_embeddings = torch.cat(img_embeddings, dim=0)
        all_txt_embeddings = classifier.T.cpu()
        all_labels = torch.tensor(labels)
        all_dataset_labels = torch.tensor(dataset_labels)

        output_path = f"dataset_embeddings_all_no_templete_{args.name}.pt"
        torch.save({"img_embeddings": all_img_embeddings,
                    "txt_embeddings": all_txt_embeddings,
                    "labels": all_labels,
                    "dataset_labels": all_dataset_labels
                    }, output_path)
        print(f"Embedding saved to {output_path}")

        # return top1, top5, torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)
    
    return top1

def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if args.distributed and not args.horovod:
        model = model.module

    #assert (len(data.keys()))==1

    if 'MedicalMNIST' in data:
        data_name = 'MedicalMNIST'
        classnames = MEDICALMNIST_CLASSNAMES 
        templates = SIMPLE_MEDICALMNIST_TEMPLATES
    elif 'PathMNIST_val' in data:
        data_name = 'PathMNIST_val'
        classnames = PATHMNIST_CLASSNAMES
        templates = SIMPLE_MEDICALMNIST_TEMPLATES
    elif "scar_val" in data:
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
            # num_classes_per_batch=6,
            device=device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    
    logging.info(f'Starting validation {data_name}.')

    top1 = run(model, classifier, data[data_name].dataloader, args, tokenizer)    
    
    #results[f'{data_name}-zeroshot-val-top1'] = top1
    #results[f'{data_name}-zeroshot-val-top5'] = top5

    logging.info(f'Finished validation {data_name}.')

    return top1, classifier


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