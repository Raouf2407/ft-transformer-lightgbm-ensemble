import torch
import torch.nn as nn

class ReGLU(nn.Module):
    # activation utilisee dans le papier FT-Transformer, marche mieux que ReLU ici
    def forward(self, x):
        x_lin, x_gate = x.chunk(2, dim=-1)
        return x_lin * torch.relu(x_gate)

class FeatureTokenizer(nn.Module):
    """Transforme chaque colonne en un token de dimension d_emb."""
    def __init__(self, n_num, cat_dims, d_emb):
        super().__init__()
        # une feature numerique = valeur * poids + biais 
        self.num_weights = nn.Parameter(torch.Tensor(n_num, d_emb))
        self.num_biases = nn.Parameter(torch.Tensor(n_num, d_emb))
        nn.init.xavier_uniform_(self.num_weights)
        nn.init.zeros_(self.num_biases)
        # une table d'embedding par colonne categorielle
        self.cat_embeddings = nn.ModuleList([nn.Embedding(d, d_emb) for d in cat_dims])
        self.cat_biases = nn.ParameterList([nn.Parameter(torch.zeros(d_emb)) for d in cat_dims])
        for emb in self.cat_embeddings:
            nn.init.xavier_uniform_(emb.weight)
        # token CLS comme dans BERT, c'est lui qui portera la prediction
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, d_emb))
        nn.init.normal_(self.cls_token)

    def forward(self, x_num, x_cat):
        batch = x_num.shape[0]
        tokens = []
        if x_num.shape[1] > 0:
            x_num = x_num.unsqueeze(2)
            tokens.append(x_num * self.num_weights.unsqueeze(0) + self.num_biases.unsqueeze(0))

        if len(self.cat_embeddings) > 0:
            cat_tokens = [(emb(x_cat[:, i]) + bias).unsqueeze(1) for i, (emb, bias) in enumerate(zip(self.cat_embeddings, self.cat_biases))]
            tokens.append(torch.cat(cat_tokens, dim=1))
        x = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(batch, -1, -1)
        return torch.cat([cls, x], dim=1)

class TransformerBlock(nn.Module):
    def __init__(self, d_emb, n_heads, ffn_factor=4 / 3, attn_dropout=0.2, ffn_dropout=0.1, resid_dropout=0.0):
        super().__init__()
        self.norm_attn= nn.LayerNorm(d_emb)
        self.attn= nn.MultiheadAttention(d_emb, n_heads, dropout=attn_dropout, batch_first=True)
        self.drop_attn= nn.Dropout(resid_dropout)
        self.norm_ffn= nn.LayerNorm(d_emb)
        d_inner= int(d_emb * ffn_factor)
        self.ffn= nn.Sequential(nn.Linear(d_emb, d_inner * 2), ReGLU(), nn.Dropout(ffn_dropout), nn.Linear(d_inner, d_emb))
        self.drop_ffn= nn.Dropout(resid_dropout)

    def forward(self, x):
        # pre-norm : la normalisation avant le bloc, ca stabilise l'entrainement
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.drop_attn(attn_out)
        h = self.norm_ffn(x)
        x = x + self.drop_ffn(self.ffn(h))
        return x

class FTTransformer(nn.Module):
    def __init__(self, n_num, cat_dims, d_emb=192, n_layers=3, n_heads=8, n_targets=3,
                 ffn_factor=4 / 3, attn_dropout=0.2, ffn_dropout=0.1, resid_dropout=0.0):
        super().__init__()
        self.tokenizer= FeatureTokenizer(n_num, cat_dims, d_emb)
        self.blocks= nn.ModuleList([TransformerBlock(d_emb, n_heads, ffn_factor, attn_dropout,ffn_dropout, resid_dropout) for i in range(n_layers)])
        self.norm_final= nn.LayerNorm(d_emb)
        self.head= nn.Linear(d_emb, n_targets)  # 3 sorties : satisfaction, wip, investissement

    def forward(self, x_num, x_cat):
        x= self.tokenizer(x_num, x_cat)
        for block in self.blocks:
            x= block(x)
        cls= self.norm_final(x[:, 0, :])  # on ne garde que le token CLS
        return self.head(cls)
