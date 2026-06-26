# Splasher

Outil de **labélisation** dont le cœur est générique : on lui donne un *dataset
synchrone* — à chaque instant, un **pack de canaux nommés** (nuage de points 3D,
image caméra, pose, …) — et on labélise soit une **grille 2D vue-de-dessus (BEV)**,
soit directement les **points 3D**, soit les deux.

Premier cas d'usage : la **traversabilité**. Mais rien n'est câblé en dur : pas de
schéma de classes imposé, pas de sémantique monde imposée, et **aucune dépendance
obligatoire à un format de dataset**. apairo n'est qu'un adaptateur d'entrée optionnel.

## Idée

- Plusieurs canaux synchronisés, affichés comme références : on se balade librement
  dans le nuage 3D, on regarde les images caméra.
- On choisit les **canaux** à afficher (dock *Canaux* : montrer/masquer chaque nuage
  ou caméra disponible dans la source — plusieurs caméras et nuages possibles).
- On **dessine la grille de carrés** (vue de dessus) : son étendue et la taille de
  chaque carré, créée explicitement via **« Nouvelle grille »**. L'annulation (undo)
  est **par frame**.
- On **sélectionne un rectangle** à la souris sur cette vue de dessus. Selon la cible :
  - **Grid** : remplit les carrés couverts de la classe active (sortie = raster d'IDs).
  - **Points** : assigne la classe aux points 3D dans le rectangle (sortie = labels par point).
- Mode **Sélection** (façon bureau) : tracer un rectangle sélectionne des cellules
  (**Shift** = ajouter, sélections non contiguës possibles), puis on **applique** la classe
  à toute la sélection d'un coup. Changer de grille demande **confirmation** si une
  labélisation existe déjà.
- **Cumul** : on peut cumuler ±N frames **recalées par leurs poses** dans le repère du
  frame courant (nuage plus dense pour mieux labéliser). La grille et les labels restent
  **par frame** : un coup de pinceau sur le nuage cumulé est **décumulé** vers chaque
  frame source. (Nécessite un canal `POSE`.)

## Installation

```bash
cd ~/dev/splasher
uv sync                 # cœur seul (numpy + Qt)
uv sync --extra apairo  # + adaptateur apairo (optionnel)
```

## Démo (zéro donnée externe)

```bash
uv run python examples/demo_arraysource.py
```

## Entrée

Le cœur consomme une `Source` : `__len__`, `__getitem__(i) -> Frame`, `channels()`.
`ArraySource` en construit une depuis des tableaux numpy en mémoire. `ApairoSource`
(extra `apairo`) enveloppe tout dataset apairo synchrone.
