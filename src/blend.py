import argparse
import os
import numpy as np
import pandas as pd
from data import TARGETS
from metrics import hit_rate

def charger(fichiers, noms, test=False):
    """Fusionne les predictions de plusieurs modeles sur la colonne id."""
    assert len(fichiers) == len(noms), "il faut autant de noms que de fichiers"
    out= None
    for f, nom in zip(fichiers, noms):
        df= pd.read_csv(f).sort_values("id").reset_index(drop=True)
        if test:
            df= df.rename(columns={t: f"pred_{t}_{nom}" for t in TARGETS})
            garder = ["id"] + [f"pred_{t}_{nom}" for t in TARGETS]
        else:
            df= df.rename(columns={f"pred_{t}": f"pred_{t}_{nom}" for t in TARGETS})
            garder= ["id"] + [f"pred_{t}_{nom}" for t in TARGETS]
            if out is None:
                garder += [f"true_{t}" for t in TARGETS]
        out= df[garder] if out is None else out.merge(df[garder], on="id")
    return out

def chercher_poids(X, y_true, n_essais=10000, delta=0.05, seed=42):
    """Random search sur les poids. Dirichlet = poids positifs qui somment a 1."""
    rng = np.random.default_rng(seed)
    poids = rng.dirichlet(np.ones(X.shape[1]), size=n_essais)
    # tout en une multiplication matricielle, sinon c'est beaucoup trop lent
    preds = X @ poids.T
    scores = np.mean(np.abs(preds - y_true[:, None]) <= delta, axis=0)
    best = int(np.argmax(scores))
    return poids[best], scores[best]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val", nargs="+", required=True, help="fichiers de prediction validation")
    p.add_argument("--test", nargs="+", required=True, help="fichiers de prediction test")
    p.add_argument("--noms", nargs="+", required=True, help="un nom par modele")
    p.add_argument("--essais", type=int, default=10000)
    p.add_argument("--out", default="outputs/submission_blend.csv")
    args = p.parse_args()
    if not (len(args.val) == len(args.test) == len(args.noms)):
        raise ValueError("val, test et noms doivent avoir la meme longueur")
    meta_val = charger(args.val, args.noms, test=False)
    meta_test = charger(args.test, args.noms, test=True)
    print("validation :", meta_val.shape)
    soumission = pd.DataFrame({"id": meta_test["id"]})
    for t in TARGETS:
        cols = [f"pred_{t}_{n}" for n in args.noms]
        poids, score = chercher_poids(meta_val[cols].values, meta_val[f"true_{t}"].values, args.essais)
        print(f"\n{t} -> hit rate {score:.4f}")
        for n, w in zip(args.noms, poids):
            print(f"   {n}: {w:.4f}")
        soumission[t] = meta_test[cols].values @ poids
        # comparaison avec le meilleur modele seul, pour verifier que le blend sert a qqch
        seuls = [hit_rate(meta_val[f"true_{t}"], meta_val[c]) for c in cols]
        print(f"   (meilleur modele seul : {max(seuls):.4f})")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    soumission.to_csv(args.out, index=False)
    print(f"\nsoumission ecrite : {args.out}")

if __name__ == "__main__":
    main()
