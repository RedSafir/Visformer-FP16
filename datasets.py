# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class CachedImageFolder(datasets.ImageFolder):
    """
    Wrapper ImageFolder yang memuat seluruh file gambar ke RAM (Memory Cache).
    Memotong latency I/O disk menjadi 0ms pada epoch 2 dan seterusnya.
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        print(f"[*] Memuat {len(self.samples):,} gambar ke RAM Cache...")
        self.cache = []
        for path, target in self.samples:
            sample = self.loader(path)
            self.cache.append((sample, target))
        print(f"[✓] Berhasil menyimpan {len(self.cache):,} gambar di RAM Cache!")

    def __getitem__(self, index):
        sample, target = self.cache[index]
        if self.transform is not None:
            sample = self.transform(sample.copy())
        return sample, target


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    use_cache = getattr(args, 'cache_ram', False)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set in ['IMNET', 'IMNET100', 'IMNET10']:
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if use_cache:
            dataset = CachedImageFolder(root, transform=transform)
        else:
            dataset = datasets.ImageFolder(root, transform=transform)
        
        nb_classes = 1000 if args.data_set == 'IMNET' else (100 if args.data_set == 'IMNET100' else 10)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        if args.std_aug:
            normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            transform = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(
                    args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)