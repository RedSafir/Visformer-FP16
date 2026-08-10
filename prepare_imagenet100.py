"""
Skrip Ekstraksi Dataset ImageNet-100 dari Hugging Face ke Struktur Folder ImageFolder (train/ dan val/).
"""

import os
import sys

# Hapus path direktori lokal agar 'import datasets' mengimpor pustaka Hugging Face datasets, bukan datasets.py lokal
if sys.path[0] == '' or sys.path[0] == os.getcwd():
    sys.path.pop(0)

try:
    from datasets import load_dataset
except ImportError:
    print("Pustaka 'datasets' belum terinstal. Silakan jalankan: pip install datasets pillow")
    sys.exit(1)

def main():
    print("=== Ekstraksi ImageNet-100 dari Hugging Face ===")
    print("Memuat dataset dari cache / Hugging Face...")
    ds = load_dataset('clane9/imagenet-100')

    output_dir = './imagenet100_data'

    for split in ['train', 'validation']:
        target_split = 'val' if split == 'validation' else split
        print(f"\nMengekstrak split '{target_split}'...")
        split_data = ds[split]
        total_items = len(split_data)

        for idx, item in enumerate(split_data):
            label = item['label']
            folder = os.path.join(output_dir, target_split, f'class_{label:03d}')
            os.makedirs(folder, exist_ok=True)
            
            img = item['image'].convert('RGB')
            img.save(os.path.join(folder, f'img_{idx}.jpg'))

            if (idx + 1) % 5000 == 0 or (idx + 1) == total_items:
                print(f"  Progres {target_split}: {idx + 1}/{total_items} gambar tersimpan.")

    print(f"\n[✓] Ekstraksi selesai! Folder '{output_dir}' siap digunakan untuk pelatihan.")

if __name__ == '__main__':
    main()
