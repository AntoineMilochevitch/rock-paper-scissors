# Rock Paper Scissors Dataset Project

Ce projet sert à télécharger, analyser et préparer un dataset d'images pour entraîner un CNN de classification entre trois classes: `paper`, `rock` et `scissors`.

Le dataset source est organisé en trois splits: `train`, `test` et `validation`. Le projet fournit ensuite un dataset préparé avec:

- un split `validation` rééquilibré à partir du `train` et de la validation existante;
- un prétraitement des images en taille fixe;
- des augmentations sur le `train` uniquement;
- des graphiques d'analyse et de comparaison avant/après.

## Structure du projet

- `src/import_dataset.py`
  - Télécharge le dataset Kaggle via `kagglehub`.
  - Copie le dataset dans `data/rock-paper-scissors-dataset`.

- `src/analyze_dataset.py`
  - Analyse le dataset brut.
  - Génère des graphiques dans `reports/dataset_analysis`.
  - Produit notamment des vues sur les tailles, les pixels, la luminosité et les couleurs.

- `src/prepare_dataset.py`
  - Construit un dataset préparé dans `data/rock-paper-scissors-prepared`.
  - Rééquilibre `validation`.
  - Applique le prétraitement et les augmentations sur `train`.
  - Génère des graphiques avant/après dans `reports/dataset_preparation`.

- `data/`
  - Contient les datasets téléchargés et préparés.

- `reports/`
  - Contient les figures produites par l'analyse et la préparation.

## Prérequis

- Python 3.10 ou plus récent.
- Un environnement virtuel `venv`.
- Une connexion Internet pour télécharger le dataset.

## Installation sur Linux

Depuis la racine du projet:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Installation sur Windows

### PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Invite de commandes

```bat
py -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Utilisation

### 1. Télécharger le dataset

```bash
python src/import_dataset.py
```

Le dataset est copié dans:

```text
data/rock-paper-scissors-dataset
```

### 2. Analyser le dataset

```bash
python src/analyze_dataset.py
```

Les graphiques sont générés dans:

```text
reports/dataset_analysis
```

### 3. Préparer le dataset pour l'entraînement

```bash
python src/prepare_dataset.py
```

Le dataset préparé est écrit dans:

```text
data/rock-paper-scissors-prepared
```

Les graphiques de comparaison avant/après sont générés dans:

```text
reports/dataset_preparation
```

## Ce que fait la préparation

Le script `prepare_dataset.py`:

- conserve `test` sans modification logique du contenu;
- construit une `validation` plus utile à partir de la validation existante et d'un échantillon stratifié du `train`;
- retire du `train` les images envoyées dans `validation` pour éviter toute duplication;
- redimensionne toutes les images à une taille fixe;
- applique des augmentations uniquement sur `train`;
- génère des graphiques pour comparer l'état avant et après préparation.

## Arguments utiles

Le script de préparation accepte quelques paramètres:

```bash
python src/prepare_dataset.py --help
```

## Modèles expérimentaux

L'objectif du projet est de comparer plusieurs approches pour classifier les trois gestes `paper`, `rock` et `scissors`.

Les modèles seront évalués avec les mêmes métriques: courbe de loss, accuracy, matrice de confusion, precision et recall. L'idée est de comparer chaque modèle à son meilleur niveau, avec early stopping et un nombre d'époques suffisant pour converger, sans en faire une variable principale de comparaison.

### 1. Modèle from scratch de base

Modèle simple servant de référence.

- architecture peu profonde;
- nombre de filtres modéré;
- kernels principalement en `3x3`;
- pooling simple et stable, typiquement `MaxPooling`;
- activation `ReLU`;
- Batch Normalization activée;
- dropout léger;
- optimiseur `Adam`.

Objectif: obtenir un bon compromis entre temps d'entraînement et performance.

### 2. Modèle from scratch plus performant

Modèle plus profond et plus capacitaire.

- plus de blocs convolutionnels;
- davantage de filtres par couche;
- kernels éventuellement plus variés, tout en restant raisonnables;
- `MaxPooling` conservé;
- Batch Normalization activée;
- dropout plus marqué que sur le modèle de base;
- optimiseur `Adam`.

Objectif: maximiser la performance, quitte à augmenter le temps d'entraînement.

### 3. Variantes from scratch ciblées

Pour l'étude des hyperparamètres, certaines variantes pourront être comparées au modèle de base:

- avec et sans Batch Normalization;
- différents niveaux de dropout;
- `Adam` vs `AdamW`;
- architecture plus ou moins profonde.

L'idée est de garder le pooling cohérent et de ne pas multiplier les différences dans un même modèle, afin d'identifier clairement l'effet de chaque choix.

### 4. Modèle en transfert learning

Un modèle préentraîné sera aussi testé pour servir de comparaison avec les modèles from scratch.

- base convolutionnelle préentraînée sur un grand dataset;
- tête de classification remplacée pour les 3 classes du projet;
- phase 1: base gelée pour entraîner uniquement la tête;
- phase 2: fine-tuning partiel si nécessaire;
- mêmes métriques d'évaluation que les modèles from scratch.

Objectif: mesurer le gain apporté par le transfert learning par rapport à un entraînement entièrement from scratch.

### 5. Ce qui reste fixe entre les modèles

Pour que la comparaison soit propre, les points suivants resteront identiques autant que possible:

- même dataset préparé;
- mêmes splits train / validation / test;
- même prétraitement de base;
- même stratégie d'early stopping;
- mêmes métriques finales;
- comparaison sur le meilleur checkpoint de validation.

Les augmentations sont déjà intégrées dans la préparation du dataset pour la partie `train`, donc elles ne seront pas re-testées comme variable principale dans cette phase.

