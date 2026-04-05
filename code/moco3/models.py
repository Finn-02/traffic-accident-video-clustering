import numpy as np
import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )
            if project_out else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
            qkv
        )

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout=dropout)
                        )
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViViT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        output_dim,
        num_frames,
        dim=128,
        depth=2,
        heads=3,
        pool="cls",
        in_channels=3,
        dim_head=64,
        dropout=0.2,
        emb_dropout=0.2,
        scale_dim=3,
    ):
        super().__init__()

        assert pool in {"cls", "mean"}, "pool type must be either cls or mean"
        assert image_size % patch_size == 0, "Image dimensions must be divisible by patch size."

        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_dim)
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, "() n d -> b t n d", b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x = x + self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, "b t n d -> (b t) n d")
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], "(b t) d -> b t d", b=b)

        cls_temporal_tokens = repeat(self.temporal_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        return self.mlp_head(x)


class _Torchvision3DBackbone(nn.Module):
    """
    외부 입력은 항상 [B, T, C, H, W].
    내부에서만 [B, C, T, H, W]로 permute.
    """
    def __init__(self, backbone_name: str, output_dim: int):
        super().__init__()

        if backbone_name == "R3D18":
            self.backbone = r3d_18(weights=None)
        elif backbone_name == "MC3_18":
            self.backbone = mc3_18(weights=None)
        elif backbone_name == "R2Plus1D":
            self.backbone = r2plus1d_18(weights=None)
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, output_dim)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input [B, T, C, H, W], got shape={tuple(x.shape)}")
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.backbone(x)


class R3D18Encoder(_Torchvision3DBackbone):
    def __init__(self, output_dim: int):
        super().__init__(backbone_name="R3D18", output_dim=output_dim)


class MC3_18Encoder(_Torchvision3DBackbone):
    def __init__(self, output_dim: int):
        super().__init__(backbone_name="MC3_18", output_dim=output_dim)


class R2Plus1DEncoder(_Torchvision3DBackbone):
    def __init__(self, output_dim: int):
        super().__init__(backbone_name="R2Plus1D", output_dim=output_dim)


class C3DEncoder(nn.Module):
    """
    경량 C3D.
    입력: [B, T, C, H, W]
    내부: [B, C, T, H, W]
    """
    def __init__(self, output_dim: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=False),

            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=False),

            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input [B, T, C, H, W], got shape={tuple(x.shape)}")
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.features(x)
        x = self.head(x)
        return x


class SqueezeExcitation3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc(scale)
        return x * scale


class X3DBlock(nn.Module):
    """
    X3D inspired lightweight bottleneck block
    """
    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1, 1), expansion: float = 2.25):
        super().__init__()
        mid_channels = max(int(out_channels * expansion), out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=False),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=mid_channels,
                bias=False,
            ),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=False),
        )

        self.se = SqueezeExcitation3D(mid_channels, reduction=4)

        self.conv3 = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

        if stride != (1, 1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = self.conv3(out)
        out = out + identity
        out = self.relu(out)
        return out


class X3DEncoder(nn.Module):
    """
    외부 의존 없는 경량 X3D-inspired encoder.
    입력: [B, T, C, H, W]
    내부: [B, C, T, H, W]
    """
    def __init__(self, output_dim: int):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(3, 24, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=False),
        )

        self.stage1 = nn.Sequential(
            X3DBlock(24, 24, stride=(1, 1, 1)),
            X3DBlock(24, 24, stride=(1, 1, 1)),
        )

        self.stage2 = nn.Sequential(
            X3DBlock(24, 48, stride=(1, 2, 2)),
            X3DBlock(48, 48, stride=(1, 1, 1)),
        )

        self.stage3 = nn.Sequential(
            X3DBlock(48, 96, stride=(2, 2, 2)),
            X3DBlock(96, 96, stride=(1, 1, 1)),
        )

        self.head = nn.Sequential(
            nn.Conv3d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(192, output_dim),
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input [B, T, C, H, W], got shape={tuple(x.shape)}")
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x


def build_model(model_name: str, image_size: int, num_frames: int, output_dim: int):
    model_name = model_name.strip()

    if model_name == "ViViT":
        return ViViT(
            image_size=image_size,
            patch_size=16,
            output_dim=output_dim,
            num_frames=num_frames,
            scale_dim=3,
            dim=128
        )

    if model_name == "R3D18":
        return R3D18Encoder(output_dim=output_dim)

    if model_name == "MC3_18":
        return MC3_18Encoder(output_dim=output_dim)

    if model_name == "R2Plus1D":
        return R2Plus1DEncoder(output_dim=output_dim)

    if model_name == "C3D":
        return C3DEncoder(output_dim=output_dim)

    if model_name == "X3D":
        return X3DEncoder(output_dim=output_dim)

    raise ValueError(
        f"Unsupported model_name: {model_name}. "
        f"Available: ViViT, R3D18, MC3_18, R2Plus1D, C3D, X3D"
    )


def count_trainable_params_m(model: nn.Module) -> float:
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum(np.prod(p.size()) for p in parameters) / 1_000_000


if __name__ == "__main__":
    img = torch.ones([2, 16, 3, 224, 224])

    for model_name in ["ViViT", "R3D18", "MC3_18", "R2Plus1D", "C3D", "X3D"]:
        model = build_model(model_name, image_size=224, num_frames=16, output_dim=4)
        params_m = count_trainable_params_m(model)
        out = model(img)
        print(f"{model_name} | Trainable Parameters: {params_m:.3f}M | Output shape: {tuple(out.shape)}")