import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from metrics import focused_loss, hit_rate, hit_rate_lgbm

def test_hit_rate_parfait():
    y= np.array([0.1, 0.2, 0.3])
    assert hit_rate(y, y) == 1.0

def test_hit_rate_limite():
    # pile a 0.05 ca compte comme un hit, juste au dessus non
    assert hit_rate([0.0], [0.05])== 1.0
    assert hit_rate([0.0], [0.051])== 0.0

def test_hit_rate_moitie():
    assert hit_rate([0.0, 0.0], [0.01, 0.5])== 0.5

def test_format_lgbm():
    nom,val,plus_grand_mieux= hit_rate_lgbm(np.array([0.1]), np.array([0.1]))
    assert nom== "hit_rate"
    assert val== 1.0
    assert plus_grand_mieux is True

def test_focused_loss_dans_la_marge():
    # une erreur de 0.01 est sous la tolerance, il ne reste que le terme MAE
    pred= torch.tensor([0.51])
    vrai= torch.tensor([0.50])
    assert focused_loss(pred, vrai).item() < 0.001


def test_focused_loss_penalise_les_gros_ecarts():
    petite= focused_loss(torch.tensor([0.55]), torch.tensor([0.50]))
    grosse= focused_loss(torch.tensor([0.90]), torch.tensor([0.50]))
    assert grosse > petite
