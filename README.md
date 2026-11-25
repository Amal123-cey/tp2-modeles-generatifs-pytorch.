# Activité Pratique N°2 — Modèles génératifs avec PyTorch

**Enseignant :** Mohamed YOUSSFI  
**Matière :** IA Générative  
**Année universitaire :** 2026–2027  
**Nom :** À compléter  
**Prénom :** À compléter  
**Groupe :** À compléter

**Lien GitHub :**

Ce travail compare, sur MNIST, un modèle discriminatif et deux modèles génératifs :

- un classifieur entièrement connecté, entraîné 3 époques ;
- un autoencodeur classique, dimension latente 2, entraîné 5 époques ;
- un VAE, dimension latente 2, entraîné 20 époques avec \(\beta = 1\).

Les architectures, les pertes et les hyperparamètres suivent le notebook du cours. Les parties GAN et diffusion ne font pas partie de ce livrable. Le notebook à lire est `notebooks/TP2_Modeles_Generatifs_Complet.ipynb`.

## Documents de référence

- `Modeles_generatifs_pytorch_avec_explications.ipynb` : architectures, pertes et ordre des expériences. Ce fichier n'est pas modifié.

## Modèles

| Modèle | Fichier de poids | Entraînement |
|---|---|---|
| Classifieur `784 → 128 → 10` | `models/classifier_mnist.pth` | Entropie croisée, Adam \(10^{-3}\), 3 époques |
| Autoencodeur `784 → 128 → 32 → 2 → 32 → 128 → 784` | `models/autoencoder_mnist.pth` | MSE, Adam \(10^{-3}\), 5 époques |
| VAE `784 → 256 → 64`, têtes `mu` et `logvar`, décodeur `2 → 64 → 256 → 784` | `models/vae_mnist.pth` | BCE + KL, \(\beta = 1\), Adam \(10^{-3}\), 20 époques |

## Architecture du projet

```text
.
├── .venv/
├── data/                          MNIST, téléchargé par TorchVision
├── models/
│   ├── classifier_mnist.pth
│   ├── autoencoder_mnist.pth
│   └── vae_mnist.pth
├── results/
│   ├── metriques.json
│   ├── comparaison_metriques.csv
│   ├── mnist_exemples.png
│   ├── comparaison_ae_vae.png
│   ├── classifier/
│   ├── autoencoder/
│   └── vae/
├── notebooks/
│   └── TP2_Modeles_Generatifs_Complet.ipynb
├── Modeles_generatifs_pytorch_avec_explications.ipynb
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore
```

## Dépendances

Python 3.11, géré par uv. PyTorch **2.15.0+cpu** et TorchVision **0.29.2+cpu** viennent de l'index CPU officiel. TorchEval **0.0.7** est compatible avec cette version. Matplotlib, NumPy, scikit-learn, pandas, tqdm, ipykernel, nbconvert et Jupyter viennent de PyPI. Les versions exactes sont dans `uv.lock`.

## Installation

À la racine du projet, dans PowerShell :

```powershell
uv sync
```

L'activation manuelle est :

```powershell
.\.venv\Scripts\Activate.ps1
```

Si la politique d'exécution bloque l'activation, les commandes `uv run` suffisent.

## Ouvrir le notebook dans VS Code

1. Ouvrir le dossier du projet.
2. Sélectionner l'interpréteur `.venv\Scripts\python.exe`.
3. Enregistrer le noyau si nécessaire :

```powershell
uv run python -m ipykernel install --sys-prefix --name tp2-modeles-generatifs --display-name "Python 3.11 (TP2 Modeles Generatifs)"
```

4. Ouvrir `notebooks/TP2_Modeles_Generatifs_Complet.ipynb`.
5. Choisir le noyau **Python 3.11 (TP2 Modeles Generatifs)**.

## Lancement

Le notebook peut être exécuté cellule par cellule. Pour tout relancer :

```powershell
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/TP2_Modeles_Generatifs_Complet.ipynb
```

Le calcul utilise CUDA si une carte NVIDIA est disponible, sinon le CPU. La graine aléatoire est 42. MNIST est téléchargé dans `data/` au premier lancement.

## Résultats de l'exécution

Entraînement complet sur CPU. Les valeurs viennent de `results/metriques.json`.

| Mesure | Valeur |
|---|---|
| Accuracy du classifieur | 96,31 % (9 631 / 10 000) |
| MSE test de l'autoencodeur | 0,0432 par pixel |
| BCE test du VAE | 135,73 par image |
| KL test du VAE | 6,51 par image |
| Époques | 3, 5 et 20 |

La MSE et la BCE ne se comparent pas directement. Sur une MSE par pixel commune, avec \(z = \mu\) pour le VAE, cette exécution donne 0,0432 pour l'autoencodeur et 0,0374 pour le VAE.

## Fichiers de résultats

- `results/metriques.json` : accuracy, MSE, BCE, KL et nombre d'époques.
- `results/comparaison_metriques.csv` : le même tableau, en CSV.
- `results/classifier/` : courbe de perte, matrice de confusion, réussites et erreurs.
- `results/autoencoder/` : perte, reconstructions, espace latent, codes moyens des chiffres 1, 3 et 8, échantillons non régularisés.
- `results/vae/` : pertes, reparamétrisation, reconstructions, espace latent, 20 images générées, grille 15 × 15, interpolation.
- `results/comparaison_ae_vae.png` : originales, reconstructions de l'autoencodeur et reconstructions du VAE.

## Limites connues

- L'espace latent a deux dimensions. Les reconstructions et les images générées restent grossières.
- Les réseaux sont entièrement connectés. Ils n'exploitent pas la structure spatiale des images.
- Le VAE génère sans étiquette : on ne choisit pas le chiffre produit.
- La MSE de l'autoencodeur et la BCE du VAE ne mesurent pas la même chose.
- Un vecteur latent aléatoire décodé par l'autoencodeur n'est pas une génération fiable.
- Les parties GAN et diffusion ne sont pas implémentées.
