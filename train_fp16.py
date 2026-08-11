"""
Skrip Pelatihan VisFormer Murni FP16 pada Dataset ImageNet-100 (atau ImageNet/CIFAR).

Cara Menjalankan:
python train_fp16.py --data-path /path/to/imagenet100 --model visformer_tiny_fp16 --batch-size 64 --epochs 100
"""

import argparse
import time
import os
import datetime
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from datasets import build_dataset
from visformer_fp16 import visformer_tiny_fp16, visformer_small_fp16
from resnet_fp16 import resnet18_fp16, resnet32_fp16, resnet32_cifar_fp16, resnet56_fp16


def get_args():
    parser = argparse.ArgumentParser('Pelatihan Model FP16 pada ImageNet-100', add_help=True)
    parser.add_argument('--data-path', default='./imagenet100', type=str,
                        help='Path ke direktori dataset (berisi folder train/ dan val/)')
    parser.add_argument('--data-set', default='IMNET100', type=str, choices=['IMNET100', 'IMNET', 'IMNET10', 'CIFAR'],
                        help='Nama dataset (default: IMNET100)')
    parser.add_argument('--model', default='visformer_tiny_fp16', type=str, 
                        choices=['visformer_tiny_fp16', 'visformer_small_fp16', 'resnet18_fp16', 'resnet32_fp16', 'resnet32_cifar_fp16', 'resnet56_fp16'],
                        help='Varian arsitektur model FP16 (VisFormer atau ResNet)')
    parser.add_argument('--batch-size', default=64, type=int, help='Ukuran batch per GPU')
    parser.add_argument('--epochs', default=100, type=int, help='Jumlah epoch pelatihan')
    parser.add_argument('--input-size', default=224, type=int, help='Resolusi gambar masukan (224x224)')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate awal')
    parser.add_argument('--weight-decay', default=0.05, type=float, help='Weight decay AdamW')
    parser.add_argument('--workers', default=4, type=int, help='Jumlah worker DataLoader')
    parser.add_argument('--cache-ram', action='store_true', default=False, help='Simpan seluruh dataset gambar ke RAM untuk kecepatan maksimum')
    parser.add_argument('--output-dir', default='./checkpoints_fp16', type=str, help='Direktori penyimpanan checkpoint')
    parser.add_argument('--print-freq', default=200, type=int, help='Frekuensi cetak log batch (default: setiap 200 batch)')
    
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


class DynamicLossScaler:
    """
    Manual Dynamic Loss Scaler untuk pelatihan presisi FP16 tanpa PyTorch AMP.
    Mencegah gradient underflow dan memantau ketersediaan nilai Inf/NaN pada gradien.
    """
    def __init__(self, init_scale: float = 128.0, growth_factor: float = 2.0, backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._successful_steps = 0

    def update(self, valid_grads: bool):
        """Update scale factor secara dinamis berdasarkan keberhasilan gradien."""
        if valid_grads:
            self._successful_steps += 1
            if self._successful_steps >= self.growth_interval:
                self.scale *= self.growth_factor
                self._successful_steps = 0
        else:
            self.scale = max(1.0, self.scale * self.backoff_factor)
            self._successful_steps = 0


class FP16OptimizerWrapper:
    """
    Wrapper Optimizer untuk Pelatihan Presisi 16-bit Murni (Tanpa AMP).
    Mempertahankan FP32 Master Weights untuk optimizer AdamW agar v_t tidak underflow.
    """
    def __init__(self, model: nn.Module, base_optimizer_cls, lr: float = 5e-4, weight_decay: float = 0.05):
        self.model = model
        # Master weights dalam FP32
        self.master_params = [
            p.detach().clone().float().requires_grad_() for p in model.parameters()
        ]
        self.optimizer = base_optimizer_cls(self.master_params, lr=lr, weight_decay=weight_decay)
        self.param_map = list(zip(list(model.parameters()), self.master_params))

    def zero_grad(self):
        self.optimizer.zero_grad()
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

class FP16OptimizerWrapper:
    """
    Wrapper Optimizer untuk Pelatihan Presisi 16-bit Murni (Tanpa AMP).
    Mempertahankan FP32 Master Weights untuk optimizer AdamW agar v_t tidak underflow.
    """
    def __init__(self, model: nn.Module, base_optimizer_cls, lr: float = 5e-4, weight_decay: float = 0.05):
        self.model = model
        # Master weights dalam FP32
        self.master_params = [
            p.detach().clone().float().requires_grad_() for p in model.parameters()
        ]
        self.optimizer = base_optimizer_cls(self.master_params, lr=lr, weight_decay=weight_decay)
        self.param_map = list(zip(list(model.parameters()), self.master_params))

    def zero_grad(self):
        self.optimizer.zero_grad()
        for p_model, _ in self.param_map:
            if p_model.grad is not None:
                p_model.grad.detach_()
                p_model.grad.zero_()

    def step(self, loss_scaler, max_norm: float = 1.0) -> bool:
        # Step 1: Periksa apakah ada NaN/Inf pada gradien FP16 model
        has_nan_or_inf = False
        for p_model, _ in self.param_map:
            if p_model.grad is not None:
                if torch.isnan(p_model.grad).any() or torch.isinf(p_model.grad).any():
                    has_nan_or_inf = True
                    break

        if has_nan_or_inf:
            loss_scaler.update(valid_grads=False)
            self.zero_grad()
            return False

        # Step 2: Salin & unscale gradien FP16 model ke master_params FP32 tanpa alokasi memori baru
        inv_scale = 1.0 / loss_scaler.scale
        for p_model, p_master in self.param_map:
            if p_model.grad is not None:
                if p_master.grad is None:
                    p_master.grad = torch.empty_like(p_master)
                p_master.grad.copy_(p_model.grad).mul_(inv_scale)
            else:
                p_master.grad = None

        # Step 3: Gradient Norm Clipping pada master parameters FP32
        torch.nn.utils.clip_grad_norm_(self.master_params, max_norm=max_norm)

        # Step 4: Optimizer step pada FP32 Master Weights
        self.optimizer.step()
        loss_scaler.update(valid_grads=True)

        # Step 5: Salin kembali master weights FP32 yang diperbarui ke FP16 model
        with torch.no_grad():
            for p_model, p_master in self.param_map:
                p_model.copy_(p_master.half())

        return True

    @property
    def param_groups(self):
        return self.optimizer.param_groups


def train_one_epoch(model, criterion, optimizer_wrapper, data_loader, device, epoch, total_epochs, loss_scaler, warmup_epochs=5, base_lr=5e-4, print_freq=200):
    model.train()
    start_time = time.time()
    running_loss = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    total_samples = 0
    skipped_steps = 0

    total_steps = len(data_loader)
    pbar = tqdm(enumerate(data_loader), total=total_steps, desc=f"Train Epoch {epoch+1}/{total_epochs}", leave=True)

    for step, (images, targets) in pbar:
        # Linear Warmup pada epoch-epoch awal (5 epoch pertama)
        if epoch < warmup_epochs:
            warmup_total_steps = warmup_epochs * total_steps
            current_step = epoch * total_steps + step
            lr = base_lr * (current_step + 1) / warmup_total_steps
            for param_group in optimizer_wrapper.param_groups:
                param_group['lr'] = lr

        # Transfer ke GPU dan konversi ke torch.float16 secara murni
        images = images.to(device, dtype=torch.float16, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer_wrapper.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.float(), targets)

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer_wrapper.zero_grad()
            loss_scaler.update(valid_grads=False)
            skipped_steps += 1
            pbar.set_postfix({'loss': 'NaN/Inf', 'scale': loss_scaler.scale, 'skipped': skipped_steps})
            continue

        # Dynamic Loss Scaling sebelum backward()
        scaled_loss = loss * loss_scaler.scale
        scaled_loss.backward()

        # Step optimizer dengan master weights FP32
        success = optimizer_wrapper.step(loss_scaler, max_norm=1.0)
        if not success:
            skipped_steps += 1
            pbar.set_postfix({'loss': 'GradNaN', 'scale': loss_scaler.scale, 'skipped': skipped_steps})
            continue

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = images.size(0)
        total_samples += batch_size
        running_loss += loss.item() * batch_size
        top1_acc += acc1.item() * batch_size
        top5_acc += acc5.item() * batch_size

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'top1': f"{acc1.item():.1f}%",
            'scale': f"{loss_scaler.scale:.0f}"
        })

    epoch_time = time.time() - start_time
    if skipped_steps > 0:
        print(f"[INFO] Total batch dilewati pada Epoch {epoch+1} karena NaN/Inf: {skipped_steps}")

    if total_samples == 0:
        return 0.0, 0.0, 0.0, epoch_time, skipped_steps

    return running_loss / total_samples, top1_acc / total_samples, top5_acc / total_samples, epoch_time, skipped_steps


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    running_loss = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    total_samples = 0

    pbar = tqdm(data_loader, desc="Evaluating", leave=False)
    for images, targets in pbar:
        images = images.to(device, dtype=torch.float16, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs.float(), targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = images.size(0)
        total_samples += batch_size
        running_loss += loss.item() * batch_size
        top1_acc += acc1.item() * batch_size
        top5_acc += acc5.item() * batch_size

        pbar.set_postfix({
            'val_loss': f"{loss.item():.4f}",
            'val_top1': f"{acc1.item():.1f}%"
        })

    return running_loss / total_samples, top1_acc / total_samples, top5_acc / total_samples


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Optimisasi kecepatan konvolusi GPU
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

    # 2. Build FP16 Model
    is_cifar = (args.data_set == 'CIFAR')
    if args.model == 'visformer_tiny_fp16':
        model = visformer_tiny_fp16(num_classes=nb_classes).to(device)
    elif args.model == 'visformer_small_fp16':
        model = visformer_small_fp16(num_classes=nb_classes).to(device)
    elif args.model == 'resnet18_fp16':
        model = resnet18_fp16(num_classes=nb_classes, is_cifar=is_cifar).to(device)
    elif args.model == 'resnet32_fp16':
        model = resnet32_fp16(num_classes=nb_classes, is_cifar=is_cifar).to(device)
    elif args.model == 'resnet32_cifar_fp16':
        model = resnet32_cifar_fp16(num_classes=nb_classes).to(device)
    elif args.model == 'resnet56_fp16':
        model = resnet56_fp16(num_classes=nb_classes, is_cifar=is_cifar).to(device)
    else:
        raise ValueError(f"Model tidak dikenal: {args.model}")

    print(f"Total Parameter: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Optimizer, Loss Function, & Manual Dynamic Loss Scaler
    optimizer_wrapper = FP16OptimizerWrapper(model, torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_wrapper.optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    loss_scaler = DynamicLossScaler(init_scale=128.0)

    best_acc1 = 0.0
    history_log = {
        "config": vars(args),
        "history": []
    }
    json_save_path = os.path.join(args.output_dir, f"{args.model}_history.json")

    # 4. Training Loop
    for epoch in range(args.epochs):
        current_lr = optimizer_wrapper.param_groups[0]['lr']
        print(f"\n--- Epoch {epoch+1}/{args.epochs} --- (LR: {current_lr:.6f})")
        train_loss, train_acc1, train_acc5, epoch_time, skipped_steps = train_one_epoch(
            model, criterion, optimizer_wrapper, train_loader, device, epoch, args.epochs, loss_scaler, warmup_epochs=5, base_lr=args.lr, print_freq=args.print_freq
        )
        scheduler.step()

        val_loss, val_acc1, val_acc5 = evaluate(model, criterion, val_loader, device)

        print(f"Hasil Epoch {epoch+1} ({epoch_time:.1f}s):")
        print(f"  Train -> Loss: {train_loss:.4f} | Top-1: {train_acc1:.2f}% | Top-5: {train_acc5:.2f}%")
        print(f"  Val   -> Loss: {val_loss:.4f} | Top-1: {val_acc1:.2f}% | Top-5: {val_acc5:.2f}%")

        # Record epoch metrics for JSON analysis
        epoch_metrics = {
            "epoch": epoch + 1,
            "lr": round(float(current_lr), 7),
            "train_loss": round(float(train_loss), 4),
            "train_top1": round(float(train_acc1), 2),
            "train_top5": round(float(train_acc5), 2),
            "val_loss": round(float(val_loss), 4),
            "val_top1": round(float(val_acc1), 2),
            "val_top5": round(float(val_acc5), 2),
            "epoch_time_sec": round(float(epoch_time), 1),
            "skipped_steps": int(skipped_steps)
        }
        history_log["history"].append(epoch_metrics)

        # Simpan/update file JSON setiap epoch selesai
        with open(json_save_path, 'w') as f:
            json.dump(history_log, f, indent=2)
        print(f"  [✓] Log riwayat pelatihan diperbarui di: {json_save_path}")

        # Simpan checkpoint terbaik
        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            save_path = os.path.join(args.output_dir, f"{args.model}_imagenet100_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_wrapper.optimizer.state_dict(),
                'best_acc1': best_acc1,
            }, save_path)
            print(f"  [✓] Checkpoint terbaik disimpan di: {save_path} (Val Top-1: {best_acc1:.2f}%)")

    print(f"\n=== Pelatihan Selesai! Top-1 Akurasi Terbaik: {best_acc1:.2f}% ===")
    print(f"File log JSON lengkap untuk analisis: {json_save_path}")


if __name__ == '__main__':
    main()
