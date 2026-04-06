import torch
import torch.nn as nn

class FeatureTokenizer(nn.Module):
    def __init__(self, n_num_features, cat_cardinalities, embedding_dim=64):
        super().__init__()
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities
        if n_num_features > 0:
            self.num_linear = nn.Linear(n_num_features, embedding_dim)
        else:
            self.num_linear = None
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, embedding_dim) for card in cat_cardinalities
        ])
    def forward(self, x_num, x_cat):
        tokens = []
        if self.num_linear is not None:
            tokens.append(self.num_linear(x_num))
        for i, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(x_cat[:, i].long()))
        return torch.stack(tokens, dim=1)

class FTTransformer(nn.Module):
    def __init__(self, n_num_features, cat_cardinalities, embedding_dim=64,
                 n_blocks=3, n_heads=8, ff_dim=128, dropout=0.1, output_dim=1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num_features, cat_cardinalities, embedding_dim)
        n_tokens = (1 if n_num_features > 0 else 0) + len(cat_cardinalities)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.output_layer = nn.Linear(embedding_dim, output_dim)
    def forward(self, x_num, x_cat):
        tokens = self.tokenizer(x_num, x_cat)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.transformer(tokens)
        cls_out = tokens[:, 0, :]
        return self.output_layer(cls_out).squeeze(-1)
