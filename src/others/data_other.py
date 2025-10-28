import torch
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler
from open_clip_train.data import DataInfo, get_dataset_fn
from .dataloader_other import PathMNISTDataset, ScarDataset


def get_MedicalMNIST(args, preprocess_fns, split):
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns
    
    if is_train:
        data_path = args.train_data
        preprocess_fn = preprocess_train
    else:
        data_path = args.val_data
        preprocess_fn = preprocess_val
    assert data_path

    dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    sampler = DistributedSampler(dataset) if args.distributed and is_train else None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_pathmnist(args, preprocess_fns, split):
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns
    
    if is_train:
        data_path = args.train_data
        preprocess_fn = preprocess_train
    else:
        data_path = args.val_data
        preprocess_fn = preprocess_val
    assert data_path

    dataset = PathMNISTDataset(data_path, transform=preprocess_fn)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_scardata(args, preprocess_fns, split, tokenizer=None, shuffle=True):
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns
    
    if is_train:
        data_path = args.train_data    
        preprocess_fn = preprocess_train
    else:
        data_path = args.val_data
        preprocess_fn = preprocess_val

    dataset = ScarDataset(data_path, transform=preprocess_fn, is_train=is_train, tokenizer=tokenizer)
    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
        shuffle=(sampler is None and shuffle)
    )

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    
    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_data_other(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        if 'scar' in args.train_data: 
            data['scar_train'] = get_scardata(args, preprocess_fns, "train", tokenizer=tokenizer)
        else:
            data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
                args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        if 'MedicalMNIST' in args.val_data:
            data["MedicalMNIST"] = get_MedicalMNIST(args, preprocess_fns, "MedicalMNIST")
        elif 'PathMNIST' in args.val_data:
            data['PathMNIST_val'] = get_pathmnist(args, preprocess_fns, "val")
        elif 'scar' in args.val_data:
            data['scar_val'] = get_scardata(args, preprocess_fns, "val", tokenizer=tokenizer, shuffle=False)
        else:
            data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
                args, preprocess_val, is_train=False, tokenizer=tokenizer)

    return data
