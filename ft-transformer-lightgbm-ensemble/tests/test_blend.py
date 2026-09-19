import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from blend import chercher_poids

def test_les_poids_somment_a_un():
    X = np.random.rand(50, 3)
    y = X[:, 0]
    poids, _ = chercher_poids(X, y, n_essais=200)
    assert np.isclose(poids.sum(), 1.0)
    assert (poids >= 0).all()

def test_trouve_le_bon_modele():
    # le modele 0 est parfait, le modele 1 est nul : le poids doit partir sur le 0
    y = np.linspace(0, 1, 100)
    X = np.column_stack([y, y + 10])
    poids, score = chercher_poids(X, y, n_essais=5000)
    assert poids[0] > poids[1]
    assert score > 0.5
