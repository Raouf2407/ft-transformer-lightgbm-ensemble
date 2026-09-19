import argparse
import ast
import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from data import TARGETS, load_data, make_split, prepare_ft, split_xy
from ft_transformer import FTTransformer
from metrics import hit_rate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def charger_features(chemin, colonnes):
    if not chemin or not os.path.exists(chemin):
        print("pas de liste de features, on garde tout")
        return None
    with open(chemin, encoding="utf-8") as f:
        liste = ast.literal_eval(f.read())
    gardees = [c for c in liste if c in colonnes]
    print(f"features demandees : {len(liste)} | trouvees : {len(gardees)}")
    if len(gardees) < len(liste):
        print("attention : certaines features du fichier ne sont pas dans le dataset")
    return gardees

def faire_loaders(X, y_scaled, train_idx, val_idx, num_cols, cat_cols, batch_size):
    Xn = torch.tensor(X[num_cols].values, dtype=torch.float32)
    Xc = torch.tensor(X[cat_cols].values, dtype=torch.long)
    yt = torch.tensor(y_scaled, dtype=torch.float32)
    train = TensorDataset(Xn[train_idx], Xc[train_idx], yt[train_idx])
    val = TensorDataset(Xn[val_idx], Xc[val_idx], yt[val_idx])
    return (DataLoader(train, batch_size=batch_size, shuffle=True),DataLoader(val, batch_size=batch_size, shuffle=False))

def predire(model, loader, scaler_y):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            xn, xc = batch[0].to(DEVICE), batch[1].to(DEVICE)
            preds.append(model(xn, xc).cpu().numpy())
    return scaler_y.inverse_transform(np.concatenate(preds))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.zip")
    p.add_argument("--out", default="outputs")
    p.add_argument("--features", default="outputs/top_features.txt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--d-emb", type=int, default=192)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    print("device :", DEVICE)
    train_df, test_df = load_data(args.data)
    X_all, y, _ = split_xy(train_df, test_df)
    features = charger_features(args.features, X_all.columns)
    X, y, X_test = split_xy(train_df, test_df, features)
    train_idx, val_idx = make_split(len(X))
    X, X_test, num_cols, cat_cols, cat_dims = prepare_ft(X, X_test, train_idx)
    print(f"numeriques : {len(num_cols)} | categorielles : {len(cat_cols)}")

    # on scale aussi les cibles, sinon wip (ordre de grandeur 1e6) ecrase les deux autres
    scaler_y = StandardScaler().fit(y.iloc[train_idx])
    y_scaled = scaler_y.transform(y)
    train_loader, val_loader = faire_loaders(X, y_scaled, train_idx, val_idx, num_cols, cat_cols, args.batch_size)
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test[num_cols].values, dtype=torch.float32),
            torch.tensor(X_test[cat_cols].values, dtype=torch.long),
        ),
        batch_size=args.batch_size,
    )
    model = FTTransformer(n_num=len(num_cols), cat_dims=cat_dims, d_emb=args.d_emb,n_layers=args.layers, n_heads=args.heads, n_targets=len(TARGETS)).to(DEVICE)
    print("parametres :", sum(p.numel() for p in model.parameters()))

    # pas de weight decay sur les biais et les LayerNorm, c'est la pratique standard
    no_decay = ["bias", "LayerNorm.weight", "norm"]
    groupes = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": 5e-5},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groupes, lr=args.lr)
    criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
        epochs=args.epochs, pct_start=0.3,
    )
    idx_sat = TARGETS.index("satisfaction")
    y_val_vrai = y.iloc[val_idx]["satisfaction"].values
    best_hr, best_state, sans_amelioration = -1.0, None, 0
    historique = []
    print("\n--- entrainement ---")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for xn, xc, yb in train_loader:
            xn, xc, yb = xn.to(DEVICE), xc.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xn, xc), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * xn.size(0)
        train_loss = total / len(train_loader.dataset)
        # on suit le hit rate sur satisfaction, c'est la metrique du concours
        val_pred = predire(model, val_loader, scaler_y)
        hr = hit_rate(y_val_vrai, val_pred[:, idx_sat])
        historique.append({"epoch": epoch + 1, "train_loss": train_loss, "val_hit_rate": hr})
        print(f"epoch {epoch + 1:02d} | loss {train_loss:.4f} | val hit rate {hr:.4f}")
        if hr > best_hr:
            best_hr, best_state, sans_amelioration = hr, copy.deepcopy(model.state_dict()), 0
        else:
            sans_amelioration += 1
            if sans_amelioration >= args.patience:
                print(f"early stopping (pas mieux depuis {args.patience} epochs)")
                break
    model.load_state_dict(best_state)
    print(f"\nmeilleur hit rate : {best_hr:.4f}")
    pd.DataFrame(historique).to_csv(f"{args.out}/ft_historique.csv", index=False)
    # predictions validation -> servent d'entree au blending
    val_pred = predire(model, val_loader, scaler_y)
    df_val = pd.DataFrame({"id": train_df["id"].iloc[val_idx].values})
    for i, t in enumerate(TARGETS):
        df_val[f"pred_{t}"] = val_pred[:, i]
        df_val[f"true_{t}"] = y.iloc[val_idx][t].values
    df_val.to_csv(f"{args.out}/ft_val.csv", index=False)
    test_pred = predire(model, test_loader, scaler_y)
    df_test = pd.DataFrame({"id": test_df["id"].values})
    for i, t in enumerate(TARGETS):
        df_test[t] = test_pred[:, i]
    df_test.to_csv(f"{args.out}/ft_test.csv", index=False)
    print(f"fichiers ecrits dans {args.out}/")

if __name__ == "__main__":
    main()
