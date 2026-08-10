"""
Skrip Pelatihan VisFormer Murni FP16 pada Dataset ImageNet-100 (atau ImageNet/CIFAR).

Cara Menjalankan:
python train_fp16.py --data-path /path/to/imagenet100 --model visformer_tiny_fp16 --batch-size 64 --epochs 100
"""

import argparse
import time
import os
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import build_dataset
from visformer_fp16 import visformer_tiny_fp16, visformer_small_fp16


def get_args():
    parser = argparse.ArgumentParser('Pelatihan VisFormer FP16 pada ImageNet-100', add_help=True)
    parser.add_argument('--data-path', default='./imagenet100', type=str,
                        help='Path ke direktori dataset (berisi folder train/ dan val/)')
    parser.add_argument('--data-set', default='IMNET100', type=str, choices=['IMNET100', 'IMNET', 'IMNET10', 'CIFAR'],
                        help='Nama dataset (default: IMNET100)')
    parser.add_argument('--model', default='visformer_tiny_fp16', type=str, choices=['visformer_tiny_fp16', 'visformer_small_fp16'],
                        help='Varian arsitektur model VisFormer FP16')
    parser.add_argument('--batch-size', default=64, type=int, help='Ukuran batch per GPU')
    parser.add_argument('--epochs', default=100, type=int, help='Jumlah epoch pelatihan')
    parser.add_argument('--input-size', default=224, type=int, help='Resolusi gambar masukan (224x224)')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate awal')
    parser.add_argument('--weight-decay', default=0.05, type=float, help='Weight decay AdamW')
    parser.add_argument('--workers', default=4, type=int, help='Jumlah worker DataLoader')
    parser.add_argument('--output-dir', default='./checkpoints_fp16', type=str, help='Direktori penyimpanan checkpoint')
    
    # Dummy args agar kompatibel dengan timm / datasets.py
    parser.add_argument('--std-aug', action='store_true', default=False)
    parser.add_argument('--color-jitter', type=float, default=0.4)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--train-interpolation', type=str, default='bicubic')
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    
    return parser.parse_parse_args() if hasattr(parser, 'parse_parse_args') else parser.parse_args()


def accuracy(output, target, topk=(1, 5)):
    """Menghitung akurasi Top-k untuk prediksi."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):
    model.train()
    start_time = time.time()
    running_loss = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    total_samples = 0

    for step, (images, targets) in enumerate(data_loader):
        # Transfer ke GPU dan konversi ke torch.float16 secara murni
        images = images.to(device, dtype=torch.float16, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass murni FP16 tanpa AMP
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = images.size(0)
        total_samples += batch_size
        running_loss += loss.item() * batch_size
        top1_acc += acc1.item() * batch_size
        top5_acc += acc5.item() * batch_size

        if (step + 1) % 20 == 0 or (step + 1) == len(data_loader):
            print(f"Epoch [{epoch+1}] Batch [{step+1}/{len(data_loader)}] - "
                  f"Loss: {loss.item():.4f} | Top-1: {acc1.item():.2f}% | Top-5: {acc5.item():.2f}%")

    epoch_time = time.time() - start_time
    return running_loss / total_samples, top1_acc / total_samples, top5_acc / total_samples, epoch_time


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    running_loss = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    total_samples = 0

    for images, targets in data_loader:
        images = images.to(device, dtype=torch.float16, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = images.size(0)
        total_samples += batch_size
        running_loss += loss.item() * batch_size
        top1_acc += acc1.item() * batch_size
        top5_acc += acc5.item() * batch_size

    return running_loss / total_samples, top1_acc / total_samples, top5_acc / total_samples


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Pelatihan VisFormer FP16 ===")
    print(f"Device: {device}")
    print(f"Dataset: {args.data_set} | Path: {args.data_path}")
    print(f"Model: {args.model}")
    print(f"Batch Size: {args.batch_size} | Epochs: {args.epochs}")

    # 1. Build Dataset & DataLoaders
    print("\nMemuat dataset...")
    dataset_train, nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    train_loader = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    print(f"Jumlah sampel Train: {len(dataset_train)} | Val: {len(dataset_val)} | Kelas: {nb_classes}")

    # 2. Build VisFormer FP16 Model
    if args.model == 'visformer_tiny_fp16':
        model = visformer_tiny_fp16(num_classes=nb_classes).to(device)
    else:
        model = visformer_small_fp16(num_classes=nb_classes).to(device)

    print(f"Total Parameter: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Optimizer & Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_acc1 = 0.0

    # 4. Training Loop
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} --- (LR: {optimizer.param_groups[0]['lr']:.6f})")
        train_loss, train_acc1, train_acc5, epoch_time = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch
        )
        scheduler.step()

        val_loss, val_acc1, val_acc5 = evaluate(model, criterion, val_loader, device)

        print(f"Hasil Epoch {epoch+1} ({epoch_time:.1f}s):")
        print(f"  Train -> Loss: {train_loss:.4f} | Top-1: {train_acc1:.2f}% | Top-5: {train_acc5:.2f}%")
        print(f"  Val   -> Loss: {val_loss:.4f} | Top-1: {val_acc1:.2f}% | Top-5: {val_acc5:.2f}%")

        # Simpan checkpoint terbaik
        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            save_path = os.path.join(args.output_dir, f"{args.model}_imagenet100_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc1': best_acc1,
            }, save_path)
            print(f"  [✓] Checkpoint terbaik disimpan di: {save_path} (Val Top-1: {best_acc1:.2f}%)")

    print(f"\n=== Pelatihan Selesai! Top-1 Akurasi Terbaik: {best_acc1:.2f}% ===")


if __name__ == '__main__':
    main()
