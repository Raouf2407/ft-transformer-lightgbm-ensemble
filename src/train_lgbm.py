import argparse
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from data import TARGETS, load_data, make_split, split_xy
from metrics import hit_rate, hit_rate_lgbm
PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 128,
    "learning_rate": 0.005,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

def top_features(model, n):
    """Les n features qui contribuent le plus au gain total."""
    imp = pd.DataFrame(
        {
            "feature": model.feature_name(),
            "gain": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)
    return imp.head(n)["feature"].tolist(), imp

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.zip")
    p.add_argument("--out", default="outputs")
    p.add_argument("--rounds", type=int, default=10000)
    p.add_argument("--patience", type=int, default=500)
    p.add_argument("--top-k", type=int, default=400, help="0 = garder toutes les features")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    train_df, test_df = load_data(args.data)
    print("train:", train_df.shape, "| test:", test_df.shape)
    X, y, X_test= split_xy(train_df, test_df)
    for c in X.columns:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    train_idx, val_idx = make_split(len(X))
    features = list(X.columns)
    if args.top_k:
        print(f"\n--- passage 1 : selection des {args.top_k} meilleures features ---")
        d_train = lgb.Dataset(X.iloc[train_idx], y["satisfaction"].iloc[train_idx])
        d_val = lgb.Dataset(X.iloc[val_idx], y["satisfaction"].iloc[val_idx], reference=d_train)
        m = lgb.train(
            PARAMS, d_train, num_boost_round=args.rounds,
            valid_sets=[d_val], feval=hit_rate_lgbm,
            callbacks=[lgb.early_stopping(args.patience, verbose=False)],
        )
        features, imp = top_features(m, args.top_k)
        imp.to_csv(f"{args.out}/feature_importance.csv", index=False)
        with open(f"{args.out}/top_features.txt", "w", encoding="utf-8") as f:
            f.write(repr(features))
        print(f"{len(features)} features gardees sur {X.shape[1]}")
    X = X[features]
    X_test = X_test[features]

    val_preds = pd.DataFrame({"id": train_df["id"].iloc[val_idx].values})
    test_preds = pd.DataFrame({"id": test_df["id"].values})
    for t in TARGETS:
        val_preds[f"true_{t}"] = y[t].iloc[val_idx].values

    for t in TARGETS:
        print(f"\n--- {t} ---")
        d_train = lgb.Dataset(X.iloc[train_idx], y[t].iloc[train_idx], categorical_feature=features)
        d_val = lgb.Dataset(X.iloc[val_idx], y[t].iloc[val_idx], categorical_feature=features,
                            reference=d_train)

        feval = hit_rate_lgbm if t == "satisfaction" else None
        model = lgb.train(
            PARAMS, d_train, num_boost_round=args.rounds,
            valid_sets=[d_train, d_val], valid_names=["train", "valid"], feval=feval,
            callbacks=[lgb.early_stopping(args.patience, verbose=True)],
        )
        pv = model.predict(X.iloc[val_idx], num_iteration=model.best_iteration)
        val_preds[f"pred_{t}"] = pv
        test_preds[t] = model.predict(X_test, num_iteration=model.best_iteration)
        if t == "satisfaction":
            print(f"hit rate validation : {hit_rate(y[t].iloc[val_idx], pv):.4f}")
        else:
            print(f"MAE validation : {np.mean(np.abs(y[t].iloc[val_idx] - pv)):.4f}")
    val_preds.to_csv(os.path.join(args.out, "lgbm_val.csv"), index=False)
    test_preds.to_csv(os.path.join(args.out, "lgbm_test.csv"), index=False)
    print(f"\nfichiers ecrits dans {args.out}/")


if __name__ == "__main__":
    main()
