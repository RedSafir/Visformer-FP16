"""
VisFormer (Vision Transformer) Implementation in Pure FP16 (Half Precision) PyTorch.

Persyaratan & Fitur Utama:
1. Modus Presisi Full FP16:
   - Seluruh bobot, bias, masukan, dan tensor perantara diproses dalam `torch.float16`.
   - Tanpa mengandalkan PyTorch AMP (`torch.cuda.amp` / `torch.autocast`).
2. Arsitektur VisFormer:
   - Stem konvolusional & Stage awal berbasis Konvolusi / Depthwise Separable Conv.
   - Stage Transformer berbasis Multi-Head Attention (MHA) & Spatial-aware MLP.
   - Positional Embedding 2D yang dapat dipelajari.
3. Stabilitas Numerik FP16:
   - Epsilon yang aman ($\epsilon = 1e-3$) pada LayerNorm/BatchNorm.
   - Softmax yang stabil secara numerik dengan pengurangan max logit dan $\epsilon = 1e-6$.
   - Scale factor Attention yang disesuaikan ($head\_dim^{-0.25}$) untuk menghindari overflow logit.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'VisFormerFP16',
    'visformer_tiny_fp16',
    'visformer_small_fp16',
    'FP16LayerNorm2d',
    'FP16SafeSoftmax',
]


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic Depth / DropPath untuk regularisasi model."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class FP16SafeSoftmax(nn.Module):
    """
    Softmax yang aman untuk presisi FP16.
    Mencegah overflow/underflow dengan mengoreksi max logit
    dan menambahkan epsilon yang aman pada penyebut.
    """
    def __init__(self, dim: int = -1, eps: float = 1e-4):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Kurangi nilai maksimum untuk mencegah overflow exp() pada FP16 (max FP16 ~ 65504)
        max_val = torch.max(x, dim=self.dim, keepdim=True).values
        x_scaled = x - max_val
        exp_x = torch.exp(x_scaled)
        sum_exp = torch.sum(exp_x, dim=self.dim, keepdim=True)
        return exp_x / torch.clamp(sum_exp + self.eps, min=1e-4)


class FP16LayerNorm2d(nn.Module):
    """
    LayerNorm khusus untuk tensor 2D BCHW dalam FP16.
    Menggunakan epsilon aman (eps=1e-4) untuk mencegah underflow varians.
    """
    def __init__(self, num_channels: int, eps: float = 1e-4):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1, dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1, dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Menghitung rata-rata dan varians sepanjang dimensi channel (dim=1)
        mean = x.mean(dim=1, keepdim=True)
        var = torch.mean((x - mean) ** 2, dim=1, keepdim=True)
        var = torch.clamp(var, min=0.0)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution (Depthwise Conv 3x3 + Pointwise Conv 1x1).
    Merupakan elemen kunci VisFormer untuk efisiensi komputasi lokal.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class VisFormerMlp(nn.Module):
    """
    MLP khas VisFormer dengan opsi Depthwise Conv 3x3 di antara lapisan linier/1x1 conv
    untuk memasukkan bias induktif spasial lokal.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        drop: float = 0.0,
        group: int = 8,
        spatial_conv: bool = True
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.spatial_conv = spatial_conv

        if self.spatial_conv:
            if group < 2:
                hidden_features = in_features * 5 // 6
            else:
                hidden_features = in_features * 2

        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(drop)

        if self.spatial_conv:
            # Group / Depthwise convolution untuk informasi spasial
            groups = group if group <= hidden_features and hidden_features % group == 0 else hidden_features
            self.conv2 = nn.Conv2d(
                hidden_features, hidden_features, kernel_size=3,
                stride=1, padding=1, groups=groups, bias=False
            )
            self.act2 = nn.GELU()

        self.conv3 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop1(x)

        if self.spatial_conv:
            x = self.conv2(x)
            x = self.act2(x)

        x = self.conv3(x)
        x = self.drop2(x)
        return x


class VisFormerAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHA) untuk tensor 2D (B, C, H, W).
    Dilengkapi FP16SafeSoftmax dan qk_scale yang disesuaikan untuk stabilitas FP16.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim_ratio: float = 1.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = max(1, round(dim // num_heads * head_dim_ratio))
        self.head_dim = head_dim

        # qk_scale disesuaikan (-0.25 exponent) untuk stabilitas presisi float16
        qk_scale_factor = qk_scale if qk_scale is not None else -0.25
        self.scale = head_dim ** qk_scale_factor

        self.qkv = nn.Conv2d(dim, head_dim * num_heads * 3, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.softmax = FP16SafeSoftmax(dim=-1, eps=1e-4)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(head_dim * num_heads, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Proyeksi QKV: (B, 3 * num_heads * head_dim, H, W)
        qkv_feat = self.qkv(x)

        # Rearrange tensor ke format Attention: (3, B, num_heads, H*W, head_dim)
        total_attn_dim = self.head_dim * self.num_heads
        qkv_feat = qkv_feat.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q = qkv_feat[:, 0].transpose(-2, -1)  # (B, num_heads, H*W, head_dim)
        k = qkv_feat[:, 1].transpose(-2, -1)  # (B, num_heads, H*W, head_dim)
        v = qkv_feat[:, 2].transpose(-2, -1)  # (B, num_heads, H*W, head_dim)

        # Scaled Dot-Product Attention dalam FP16
        # Matmul (q * scale) @ k^T
        q_scaled = q * self.scale
        k_scaled = k * self.scale
        attn_scores = torch.matmul(q_scaled, k_scaled.transpose(-2, -1))  # (B, num_heads, H*W, H*W)

        # Clamping attn_scores pada range [-50.0, 50.0] agar aman dari FP16 exp overflow
        attn_scores = torch.clamp(attn_scores, min=-50.0, max=50.0)

        # Softmax stabil FP16
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.attn_drop(attn_weights)

        # Agregasi Value: (B, num_heads, H*W, head_dim)
        out = torch.matmul(attn_weights, v)

        # Reshape kembali ke format spasial 2D (B, C_attn, H, W)
        out = out.transpose(-2, -1).reshape(B, total_attn_dim, H, W)

        # Proyeksi output
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class VisFormerBlock(nn.Module):
    """
    Blok Bangunan VisFormer:
    Norm1 -> Attention (opsional) -> DropPath -> Norm2 -> VisFormerMlp -> DropPath
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim_ratio: float = 1.0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        group: int = 8,
        attn_disabled: bool = False,
        spatial_conv: bool = False
    ):
        super().__init__()
        self.attn_disabled = attn_disabled
        self.spatial_conv = spatial_conv
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if not attn_disabled:
            self.norm1 = FP16LayerNorm2d(dim, eps=1e-4)
            self.attn = VisFormerAttention(
                dim, num_heads=num_heads, head_dim_ratio=head_dim_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
            )

        self.norm2 = FP16LayerNorm2d(dim, eps=1e-4)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = VisFormerMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, drop=drop,
            group=group, spatial_conv=spatial_conv
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.attn_disabled:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedFP16(nn.Module):
    """
    Patch Embedding / Downsampling berbasis Konvolusi (Depthwise Separable / Conv Standard) dalam FP16.
    """
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int = 4, use_depthwise: bool = True):
        super().__init__()
        self.patch_size = patch_size
        if use_depthwise:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, in_chans, kernel_size=patch_size, stride=patch_size, groups=in_chans, bias=False),
                nn.Conv2d(in_chans, embed_dim, kernel_size=1, bias=False)
            )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = FP16LayerNorm2d(embed_dim, eps=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


class VisFormerFP16(nn.Module):
    """
    Arsitektur Utama VisFormer (Vision Transformer) yang Berjalan Murni dalam Presisi FP16 (Half Precision).

    Arsitektur Terbagi atas 3 Stage Utama:
    - Conv Stem & Stage 1: Pengolahan fitur spasial resolusi tinggi berbasis Konvolusi.
    - Stage 2 & Stage 3: Pengolahan fitur hierarkis berbasis Self-Attention (MHA) & Spatial MLP.
    - Positional Embedding 2D yang ditambahkan pada setiap transisi stage.
    """
    def __init__(
        self,
        img_size: int = 224,
        init_channels: int = 32,
        num_classes: int = 1000,
        embed_dim: int = 384,
        depth: list = [7, 4, 4],
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_stage: str = '011',
        spatial_conv: str = '100',
        group: int = 8,
        pos_embed: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.pos_embed = pos_embed

        if isinstance(depth, (list, tuple)):
            self.stage_num1, self.stage_num2, self.stage_num3 = depth
            total_depth = sum(depth)
        else:
            self.stage_num1 = self.stage_num3 = depth // 3
            self.stage_num2 = depth - self.stage_num1 - self.stage_num3
            total_depth = depth

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # --- Stage 0: Convolutional Stem ---
        # Mengurangi resolusi H, W sebesar 2x (contoh: 224 -> 112)
        curr_img_size = img_size // 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, init_channels, kernel_size=7, stride=2, padding=3, bias=False),
            FP16LayerNorm2d(init_channels, eps=1e-4),
            nn.GELU()
        )

        # --- Stage 1 ---
        # Patch Embed 1: 112 -> 28 (patch_size=4)
        dim1 = embed_dim // 2
        self.patch_embed1 = PatchEmbedFP16(init_channels, dim1, patch_size=4, use_depthwise=True)
        curr_img_size //= 4

        if self.pos_embed:
            self.pos_embed1 = nn.Parameter(torch.zeros(1, dim1, curr_img_size, curr_img_size, dtype=torch.float16))

        self.stage1 = nn.ModuleList([
            VisFormerBlock(
                dim=dim1, num_heads=num_heads, head_dim_ratio=0.5, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                group=group, attn_disabled=(attn_stage[0] == '0'), spatial_conv=(spatial_conv[0] == '1')
            )
            for i in range(self.stage_num1)
        ])

        # --- Stage 2 ---
        # Patch Embed 2: 28 -> 14 (patch_size=2)
        dim2 = embed_dim
        self.patch_embed2 = PatchEmbedFP16(dim1, dim2, patch_size=2, use_depthwise=True)
        curr_img_size //= 2

        if self.pos_embed:
            self.pos_embed2 = nn.Parameter(torch.zeros(1, dim2, curr_img_size, curr_img_size, dtype=torch.float16))

        start_idx2 = self.stage_num1
        self.stage2 = nn.ModuleList([
            VisFormerBlock(
                dim=dim2, num_heads=num_heads, head_dim_ratio=1.0, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[start_idx2 + i],
                group=group, attn_disabled=(attn_stage[1] == '0'), spatial_conv=(spatial_conv[1] == '1')
            )
            for i in range(self.stage_num2)
        ])

        # --- Stage 3 ---
        # Patch Embed 3: 14 -> 7 (patch_size=2)
        dim3 = embed_dim * 2
        self.patch_embed3 = PatchEmbedFP16(dim2, dim3, patch_size=2, use_depthwise=True)
        curr_img_size //= 2

        if self.pos_embed:
            self.pos_embed3 = nn.Parameter(torch.zeros(1, dim3, curr_img_size, curr_img_size, dtype=torch.float16))

        start_idx3 = self.stage_num1 + self.stage_num2
        self.stage3 = nn.ModuleList([
            VisFormerBlock(
                dim=dim3, num_heads=num_heads, head_dim_ratio=1.0, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[start_idx3 + i],
                group=group, attn_disabled=(attn_stage[2] == '0'), spatial_conv=(spatial_conv[2] == '1')
            )
            for i in range(self.stage_num3)
        ])

        # --- Head / Output Stage ---
        self.norm = FP16LayerNorm2d(dim3, eps=1e-4)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dim3, num_classes)

        # Inisialisasi bobot dan secara eksplisit konversi SELURUH parameter ke torch.float16
        self._init_weights()
        self.to(torch.float16)

    def _init_weights(self):
        """Inisialisasi bobot dengan Truncated Normal pada presisi FP16."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        if self.pos_embed:
            nn.init.trunc_normal_(self.pos_embed1, std=0.02)
            nn.init.trunc_normal_(self.pos_embed2, std=0.02)
            nn.init.trunc_normal_(self.pos_embed3, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Verifikasi dan konversi masukan ke torch.float16 secara eksplisit
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        # 0. Conv Stem Stage
        x = self.stem(x)

        # 1. Stage 1 (Conv-based / Spatial MLP)
        x = self.patch_embed1(x)
        if self.pos_embed:
            x = x + self.pos_embed1
        for block in self.stage1:
            x = block(x)

        # 2. Stage 2 (Transformer Stage 1)
        x = self.patch_embed2(x)
        if self.pos_embed:
            x = x + self.pos_embed2
        for block in self.stage2:
            x = block(x)

        # 3. Stage 3 (Transformer Stage 2)
        x = self.patch_embed3(x)
        if self.pos_embed:
            x = x + self.pos_embed3
        for block in self.stage3:
            x = block(x)

        # 4. Classification Head
        x = self.norm(x)
        x = self.global_pool(x)  # (B, dim3, 1, 1)
        x = torch.flatten(x, 1)   # (B, dim3)
        x = self.head(x)         # (B, num_classes)
        return x


def visformer_tiny_fp16(num_classes: int = 1000, **kwargs) -> VisFormerFP16:
    """Konfigurasi VisFormer-Tiny dalam presisi murni FP16."""
    return VisFormerFP16(
        init_channels=16, embed_dim=192, depth=[7, 4, 4], num_heads=3,
        mlp_ratio=4.0, group=8, attn_stage='011', spatial_conv='100',
        num_classes=num_classes, **kwargs
    )


def visformer_small_fp16(num_classes: int = 1000, **kwargs) -> VisFormerFP16:
    """Konfigurasi VisFormer-Small dalam presisi murni FP16."""
    return VisFormerFP16(
        init_channels=32, embed_dim=384, depth=[7, 4, 4], num_heads=6,
        mlp_ratio=4.0, group=8, attn_stage='011', spatial_conv='100',
        num_classes=num_classes, **kwargs
    )


if __name__ == '__main__':
    print("=== Pengujian Model VisFormer Murni FP16 ===")
    
    # 1. Inisialisasi Model VisFormer-Tiny FP16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = visformer_tiny_fp16(num_classes=10).to(device)
    
    # Pastikan model secara penuh bertipe torch.float16
    model.eval()

    # 2. Buat Input Dummy berukuran (Batch=2, Channels=3, Height=224, Width=224)
    # Eksplisit dikonversi ke torch.float16
    x_input = torch.randn(2, 3, 224, 224, device=device, dtype=torch.float16)

    print(f"Device: {device}")
    print(f"Tipe data input: {x_input.dtype}")
    print(f"Total Parameter: {sum(p.numel() for p in model.parameters()):,}")

    # Verifikasi tipe parameter model
    all_fp16 = all(p.dtype == torch.float16 for p in model.parameters())
    print(f"Apakah semua parameter bertipe float16? {all_fp16}")

    # 3. Uji Forward Pass
    with torch.no_grad():
        output = model(x_input)
    
    print(f"Bentuk Tensor Output: {output.shape}")
    print(f"Tipe Data Output: {output.dtype}")
    print(f"Apakah ada nilai NaN dalam output? {torch.isnan(output).any().item()}")
    print(f"Apakah ada nilai Inf dalam output? {torch.isinf(output).any().item()}")

    # 4. Uji Backward Pass Murni FP16 (Tanpa AMP)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    
    out_train = model(x_input)
    target = torch.randint(0, 10, (2,), device=device)
    loss_fn = nn.CrossEntropyLoss()
    
    loss = loss_fn(out_train, target)
    print(f"Loss FP16: {loss.item():.4f} (dtype: {loss.dtype})")
    
    loss.backward()
    optimizer.step()
    
    print("Backward pass & optimizer step berhasil tanpa error atau AMP!")
