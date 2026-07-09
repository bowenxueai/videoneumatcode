import torch
import torch.nn as nn
import torch.nn.functional as F

torch.cuda.set_per_process_memory_fraction(0.98, device=0)

# --- Helper: Basic Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, N, D]
        # print('transformer block input', x.shape)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        # print('transformer block input', x.shape)
        return x


# --- Simple Patch Embedding ---
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=512, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # print(self.proj)

    def forward(self, x):
        # x: [B, C, H, W]
        # print('patch embed input', x.shape)
        x = self.proj(x)  # [B, D, H', W']
        # print('proj output', x.shape)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        # print('patch embed output', x.shape)
        return x, (H, W)


# --- Main LRM Backbone ---
class SimpleLRM(nn.Module):
    def __init__(self, in_frames=17, embed_dim=64, depth=16, num_heads=8, patch_size=16, output_channels=24):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels=in_frames * 3, embed_dim=embed_dim, patch_size=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 4096, embed_dim))  # max 1024 tokens
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.output_channels = output_channels
        self.head = nn.Linear(embed_dim, patch_size * patch_size * self.output_channels)
        self.patch_size = patch_size

    def forward(self, x):
        # x: [B, 4, 3, H, W]
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        # print(x.shape)
        x, (H_p, W_p) = self.patch_embed(x)

        # print("patch_embed output height", H_p, W_p)

        N = x.shape[1]
        # print(N)
        x = x + self.pos_embed[:, :N, :]
        # print('transformer input', x.shape)
        x = self.transformer(x)
        # print('transformer output', x.shape)

        # reconstruct patches
        x = self.head(x)
        x = x.reshape(B, H_p, W_p, self.output_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, self.output_channels, H_p * self.patch_size, W_p * self.patch_size)
        return x


# --- Example usage ---
if __name__ == "__main__":
    model = SimpleLRM(embed_dim=512, patch_size=4)
    model.to('cuda')
    dummy = torch.randn(2, 17, 3, 256, 256).to('cuda')  # 2 samples, 4 frames each
    out = model(dummy)
    print(out.shape)  # [2, 3, 128, 128]

    # print(model)
