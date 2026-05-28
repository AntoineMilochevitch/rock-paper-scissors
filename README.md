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


