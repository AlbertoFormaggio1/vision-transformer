import torch, torchvision
from torch import nn

# Definition of the Patch Embedding Layer
class PatchEmbeddingLayer(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 embedding_dimension: int = 768,
                 patch_size: int = 16):
        super().__init__()

        self.convolutional_layer = nn.Conv2d(in_channels=in_channels,
                                             out_channels=embedding_dimension,
                                             kernel_size=patch_size,
                                             stride=patch_size,
                                             padding=0)

        self.flatten_layer = nn.Flatten(start_dim=-2, end_dim=-1)

    def forward(self, x):
        patched = self.convolutional_layer(x)
        flattned = self.flatten_layer(patched)
        return flattned.permute(0, 2, 1)

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim=768,
                 num_heads=12):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multi_attn_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        lnorm = self.layer_norm(x)
        multihead, _ = self.multi_attn_layer(lnorm, lnorm, lnorm)
        residual = multihead + x
        return residual

class MLPBlock(nn.Module):
    def __init__(self,
                 embedding_dim=768,
                 linear_size=3072,
                 dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.MLP = nn.Sequential(
            nn.Linear(embedding_dim, linear_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(linear_size, embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        normalized = self.layer_norm(x)
        mlp = self.MLP(normalized)
        residual = x + mlp
        return residual

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embed_dim=768,
                 num_heads=12,
                 linear_size=3072,
                 dropout=0.1):
        super().__init__()
        self.multi_head_attn = MultiHeadSelfAttentionBlock(embedding_dim=embed_dim, num_heads=num_heads)
        self.mlp = MLPBlock(embedding_dim=embed_dim, linear_size=linear_size, dropout=dropout)

    def forward(self, x):
        multi_attn = self.multi_head_attn(x)
        mlp = self.mlp(multi_attn)
        return mlp

class Classifier(nn.Module):
    def __init__(self,
                 num_classes,
                 embed_dim=768,
                 mlp_size=3072,
                 dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(embed_dim),
                                 nn.Linear(embed_dim, mlp_size),
                                 nn.Tanh(),
                                 nn.Dropout(dropout),
                                 nn.Linear(mlp_size, num_classes))

    def forward(self, x):
        return self.mlp(x[:, 0]) #Processing only the classification head to get the classification (Eq. 4 paper)

class ViTModel(nn.Module):
    def __init__(self, batch_size, img_size, num_classes, patch_size=16, num_heads=12, mlp_size=3072, dropout=0.1, layers=12):
        super().__init__()
        self.emb_dim = patch_size * patch_size * 3
        assert img_size % patch_size == 0, 'The size of the image has to be multiple of the patch size'
        self.patch_num = (img_size // patch_size) ** 2
        self.patch_layer = PatchEmbeddingLayer(in_channels=3, embedding_dimension=self.emb_dim, patch_size=patch_size)
        self.cls_embedding = nn.Parameter(torch.randn((batch_size, 1, self.emb_dim)), requires_grad=True)
        self.pos_embedding = nn.Parameter(torch.randn((batch_size, self.patch_num + 1, self.emb_dim)))
        self.encoder = nn.Sequential(*[TransformerEncoderBlock(embed_dim=self.emb_dim,
                                                               num_heads=num_heads,
                                                               linear_size=mlp_size,
                                                               dropout=dropout) for _ in range(layers)])
        self.classifier = Classifier(num_classes, self.emb_dim, mlp_size, dropout)

    def forward(self, x):
        patched_img = self.patch_layer(x)
        cls_img = torch.cat([self.cls_embedding, patched_img], dim=1)
        pos_img = cls_img + self.pos_embedding
        encoded_img = self.encoder(pos_img)
        classification = self.classifier(encoded_img)

        return classification