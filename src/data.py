import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

TARGETS = ["satisfaction", "wip", "investissement"]


def load_data(path="data/data.zip"):
    # les donnees du concours arrivent en zip, mais je gere aussi un dossier
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            noms = zf.namelist()
            f_train = [n for n in noms if n.endswith("train.csv")][0]
            f_test = [n for n in noms if n.endswith("test.csv")][0]
            train = pd.read_csv(zf.open(f_train))
            test = pd.read_csv(zf.open(f_test))
    else:
        train = pd.read_csv(f"{path}/train.csv")
        test = pd.read_csv(f"{path}/test.csv")
    return train, test


def split_xy(train_df, test_df, features=None):
    X = train_df.drop(columns=["id"] + TARGETS, errors="ignore")
    y = train_df[TARGETS]
    X_test = test_df.drop(columns=["id"], errors="ignore")

    if features is not None:
        # on garde que les features qui existent vraiment (au cas ou)
        features = [f for f in features if f in X.columns]
        X = X[features]
        X_test = X_test[features]

    return X, y, X_test


def make_split(n, test_size=0.2, seed=42):
    # je split sur les indices et pas sur le dataframe, comme ca je peux
    # recuperer les id de validation pour le blending
    idx = np.arange(n)
    return train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)


def types_colonnes(X, max_card=20):
    # tout ce qui est texte, booleen ou avec peu de valeurs uniques -> categoriel
    cat = [
        c
        for c in X.columns
        if X[c].dtype == "object" or X[c].dtype == "bool" or X[c].nunique() < max_card
    ]
    num = [c for c in X.columns if c not in cat]
    return num, cat


def prepare_ft(X, X_test, train_idx, max_card=20):
    """Encodage pour le FT-Transformer : num standardises, cat en entiers."""
    num_cols, cat_cols = types_colonnes(X, max_card)
    X = X.copy()
    X_test = X_test.copy()

    # important : on fit uniquement sur le train, sinon la validation fuit
    if num_cols:
        scaler = StandardScaler()
        scaler.fit(X.iloc[train_idx][num_cols])
        X[num_cols] = scaler.transform(X[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(X.iloc[train_idx][cat_cols])
    X[cat_cols] = enc.transform(X[cat_cols])
    X_test[cat_cols] = enc.transform(X_test[cat_cols])

    # -1 = categorie jamais vue, on decale de 1 pour que l'index 0 serve a ca
    X[cat_cols] = X[cat_cols].astype(int) + 1
    X_test[cat_cols] = X_test[cat_cols].astype(int) + 1

    cat_dims = [int(X[c].max()) + 1 for c in cat_cols]
    for i, c in enumerate(cat_cols):
        X_test[c] = X_test[c].clip(0, cat_dims[i] - 1)

    return X, X_test, num_cols, cat_cols, cat_dims
