import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ft_transformer import FTTransformer

def test_shape_de_sortie():
    m= FTTransformer(n_num=3, cat_dims=[5, 4, 6], d_emb=32, n_layers=2, n_heads=4)
    out= m(torch.randn(8, 3), torch.randint(0, 4, (8, 3)))
    assert out.shape == (8, 3)

def test_sans_feature_numerique():
    # sur les donnees du concours presque tout est categoriel, il faut que ca passe
    m = FTTransformer(n_num=0, cat_dims=[5, 4], d_emb=16, n_layers=1, n_heads=2)
    out = m(torch.randn(4, 0), torch.randint(0, 4, (4, 2)))
    assert out.shape == (4, 3)

def test_le_gradient_passe():
    m= FTTransformer(n_num=2, cat_dims=[3], d_emb=16, n_layers=1, n_heads=2)
    m(torch.randn(4, 2), torch.randint(0, 3, (4, 1))).sum().backward()
    assert m.head.weight.grad is not None
