"""Genere un faux dataset pour pouvoir tester le code sans les donnees du concours."""
import argparse
import os

import numpy as np
import pandas as pd

def generer(n, n_cat, n_num, seed):
    rng= np.random.default_rng(seed)
    df= pd.DataFrame({"id": np.arange(n)})

    for i in range(n_cat):
        df[f"cat_{i}"] = rng.integers(0, rng.integers(3, 12), n)
    for i in range(n_num):
        df[f"num_{i}"] = rng.normal(0, 1, n)
    signal = df["cat_0"] * 0.05 + df["num_0"] * 0.1 + rng.normal(0, 0.05, n)
    df["satisfaction"] = np.clip(0.5 + signal, 0, 1)
    df["wip"] = 1e6 * (1 + signal) + rng.normal(0, 5e4, n)
    df["investissement"] = 500 * (1 + signal) + rng.normal(0, 20, n)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--n-cat", type=int, default=30)
    p.add_argument("--n-num", type=int, default=5)
    p.add_argument("--out", default="data")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    train = generer(args.n_train, args.n_cat, args.n_num, seed=0)
    test = generer(args.n_test, args.n_cat, args.n_num, seed=1)
    test = test.drop(columns=["satisfaction", "wip", "investissement"])
    test["id"] += args.n_train
    train.to_csv(f"{args.out}/train.csv", index=False)
    test.to_csv(f"{args.out}/test.csv", index=False)
    print(f"train {train.shape} / test {test.shape} -> {args.out}/")


if __name__ == "__main__":
    main()
