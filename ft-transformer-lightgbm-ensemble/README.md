# Hackathon Airbus : prédiction multi-cibles sur l'atelier d'assemblage

2ᵉ place sur 12 équipes, hackathon Kaggle organisé par Airbus Helicopters à IMT Mines Alès (décembre 2025).

Le problème : à partir des paramètres d'une ligne d'assemblage d'hélicoptères, prédire en même temps **trois indicateurs** — le WIP (en-cours de production), l'investissement, et la satisfaction client. La métrique du concours était le **hit rate @ 0.05** sur la satisfaction : le pourcentage de prédictions qui tombent à moins de 0,05 de la vraie valeur.

Ma solution : un **FT-Transformer codé à la main** en PyTorch, un LightGBM, et un blending des deux avec des poids optimisés directement sur la métrique.

## Résultats

| Modèle | Hit rate @0.05 (validation) |
|---|---|
| LightGBM (toutes features) | 0,775 |
| LightGBM (top 400 features) | 0,772 |
| FT-Transformer | 0,528 |
| **Blend FT-T 0,73 + LGBM 0,27** | **0,820** |

Le blend gagne 4,5 points sur le meilleur modèle seul. C'est logique : les deux modèles se trompent sur des exemples différents, un arbre et un transformer n'apprennent pas les mêmes choses.

Sur le WIP et l'investissement je reporte la MAE et pas le hit rate, parce que ces deux cibles sont sur des échelles très différentes (WIP ≈ 10⁶) : une tolérance de 0,05 y est ininterprétable, le hit rate y vaut quasiment zéro par construction.

| Cible | MAE validation |
|---|---|
| WIP | 1 247 126 |
| Investissement | 46,7 |

## Ce qu'il y a d'intéressant dedans

**Le FT-Transformer est écrit from scratch** (`src/ft_transformer.py`), pas repris d'une librairie. Chaque colonne du tableau devient un token : les features numériques via un poids et un biais appris par colonne, les catégorielles via une table d'embedding par colonne. On ajoute un token CLS comme dans BERT, on passe le tout dans des blocs d'attention en pre-norm avec activation ReGLU, et c'est le token CLS en sortie qui porte les 3 prédictions. Une seule tête pour les trois cibles, donc le modèle partage ses représentations entre elles.

**Une loss alignée sur la métrique** (`focused_loss` dans `src/metrics.py`). Le hit rate ne récompense pas le fait d'être très précis, juste le fait d'être dans la marge de 0,05. Donc au lieu d'une MSE classique j'ai écrit une loss qui ne pénalise que ce qui dépasse la marge, avec un petit terme MAE pour garder du gradient quand tout est déjà dedans. Au final j'ai quand même soumis avec une L1Loss classique, qui donnait de meilleurs résultats sur ce dataset — la loss custom est dans le code, je la garde pour la reprendre.

**La sélection de features par gain cumulé.** Le dataset a ~7 600 colonnes. J'entraîne un premier LightGBM, je trie les features par gain, je trace la courbe cumulative et je coupe au coude : 400 features suffisent à capturer l'essentiel du gain. Ça fait perdre 0,3 point de hit rate et ça divise le temps d'entraînement par plus de dix — nécessaire vu qu'on avait la journée.

**Les poids du blend sont cherchés sur la métrique, pas sur une MSE.** Je tire 10 000 jeux de poids dans une loi de Dirichlet (garantit des poids positifs qui somment à 1), et je calcule les 10 000 hit rates en une seule multiplication matricielle. Un RidgeCV aurait optimisé l'erreur quadratique, ce qui n'est pas ce qu'on nous demandait.

## Organisation

```
src/
  data.py             chargement, split, encodage
  metrics.py          hit rate + loss custom
  ft_transformer.py   le modèle
  train_lgbm.py       entraîne LightGBM et sort les top features
  train_ft.py         entraîne le FT-Transformer
  blend.py            recherche des poids et soumission finale
  make_fake_data.py   faux dataset pour tester sans les données du concours
notebooks/            les notebooks tels qu'utilisés pendant le hackathon
tests/                pytest
```

Les notebooks dans `notebooks/` sont ceux de la journée, avec les sorties d'entraînement. Le code de `src/` est la version que j'ai remise au propre après : découpée en modules, paramétrable en ligne de commande, testée.

## Lancer le code

Les données du concours ne sont pas redistribuables, donc le repo contient un générateur de faux données pour vérifier que tout tourne :

```bash
pip install -r requirements.txt

python src/make_fake_data.py              # crée data/train.csv et data/test.csv
python src/train_lgbm.py --data data --top-k 400
python src/train_ft.py  --data data --epochs 80
python src/blend.py \
    --val  outputs/ft_val.csv  outputs/lgbm_val.csv \
    --test outputs/ft_test.csv outputs/lgbm_test.csv \
    --noms ft_transformer lightgbm
```

Avec les vraies données, il suffit de mettre l'archive dans `data/data.zip` et de laisser `--data data/data.zip` (valeur par défaut).

```bash
pytest tests/
```

## Ce que je ferais différemment

- **Les embeddings catégoriels sont une boucle Python** sur les colonnes. Sur 400 features ça passe, mais la bonne façon est une seule table d'embedding avec des offsets par colonne, en un seul appel. C'est ce qui ralentit le plus l'entraînement aujourd'hui.
- **Une seule split train/validation** (80/20). Une validation croisée en 5 folds aurait donné des poids de blend bien plus fiables — là, les poids sont optimisés sur 3 000 lignes de validation, donc probablement un peu sur-ajustés.
- **Le FT-Transformer est sous-exploité.** 0,528 contre 0,775 pour LightGBM, c'est loin de ce que le papier obtient sur du tabulaire. Pendant le hackathon j'ai entraîné une première version sur les 7 600 colonnes, ce qui donne une séquence de 7 600 tokens et une attention quadratique dessus : beaucoup trop lourd pour le temps qu'on avait. Je n'ai basculé sur les 400 features sélectionnées qu'à la fin, sans avoir le temps de retoucher les hyperparamètres derrière. C'est la première chose à reprendre.
- **Je n'ai jamais testé la loss custom à fond**, faute de temps. Comme elle colle à la métrique, il y a probablement quelque chose à y gagner.
- CatBoost était prévu comme troisième modèle, je n'ai pas eu le temps de le faire tourner.

## Stack

Python, PyTorch, LightGBM, pandas, NumPy, scikit-learn.
