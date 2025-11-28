# 🖐️ Activer la Main qui Écrit

## Vue d'ensemble

La fonctionnalité `hand_draw` ajoute une main animée qui suit le tracé pendant l'animation SVG, créant un effet de "main qui dessine" comme dans les vidéos de tableau blanc.

## Utilisation Basique

### Activation Simple

```python
from kivg import Kivg

kivg = Kivg(512, 512)

# Activer la main avec hand_draw=True
frames = kivg.draw(
    'mon_fichier.svg',
    animate=True,      # Animation requise
    hand_draw=True,    # ✅ Active la main qui écrit
    fps=30
)

# Sauvegarder l'animation
kivg.save_gif('output.gif', fps=30)
```

## Paramètres de Personnalisation

### Tous les Paramètres Disponibles

```python
frames = kivg.draw(
    'fichier.svg',
    animate=True,
    hand_draw=True,
    
    # Personnalisation de la main:
    hand_image='path/to/hand.png',  # Image personnalisée (optionnel)
    hand_scale=0.30,                # Échelle de la main (0.2-0.5)
    hand_offset=(-18, -140)         # Décalage (x, y) par rapport au trait
)
```

### Description des Paramètres

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `hand_draw` | bool | `False` | Active/désactive la main |
| `hand_image` | str | `None` | Chemin vers image PNG personnalisée |
| `hand_scale` | float | `0.30` | Taille de la main (0.2 = petit, 0.5 = grand) |
| `hand_offset` | tuple | `(-18, -140)` | Position (x, y) relative au trait - positionne la pointe du stylo |

## Exemples

### Exemple 1: Animation Simple avec Main

```python
from kivg import Kivg

kivg = Kivg(600, 600, background=(255, 255, 255, 255))

# Logo avec main qui dessine
frames = kivg.draw(
    'demo/icons/python2.svg',
    animate=True,
    fill=True,
    hand_draw=True,  # Main activée
    fps=30,
    dur=0.02
)

kivg.save_gif('python_avec_main.gif', fps=30)
```

### Exemple 2: Main Personnalisée

```python
# Main plus grande et repositionnée
frames = kivg.draw(
    'demo/icons/discord.svg',
    animate=True,
    hand_draw=True,
    hand_scale=0.35,        # Main 35% de l'original
    hand_offset=(-18, -160), # Décalée vers le haut
    fps=30
)
```

### Exemple 3: Image de Main Personnalisée

```python
# Utiliser votre propre image de main
frames = kivg.draw(
    'dessin.svg',
    animate=True,
    hand_draw=True,
    hand_image='images/ma_main.png',  # Votre image
    hand_scale=0.3,
    fps=30
)
```

## Comportement

### Quand la Main Apparaît

- ✅ **Pendant l'animation des strokes** : La main suit le tracé
- ❌ **Pendant le remplissage** : La main disparaît automatiquement
- ❌ **Après l'animation** : La main n'apparaît plus

### Position de la Main

La main est positionnée en fonction de:
1. **Point de dessin actuel** : Bout du trait en cours
2. **Offset** : Décalage `(x, y)` appliqué
   - `x < 0` : Main à gauche du trait
   - `y < 0` : Main au-dessus du trait

## Ajustements Recommandés

### Pour Différents Types de Tracés

```python
# Tracés larges et rapides
hand_scale=0.35
hand_offset=(-20, -150)

# Tracés fins et détaillés
hand_scale=0.25
hand_offset=(-15, -120)

# Texte ou écriture (défaut)
hand_scale=0.30
hand_offset=(-18, -140)  # Défaut, pointe du stylo suit le trait
```

## Format de l'Image de Main

Si vous utilisez une image personnalisée (`hand_image`):

### Requis
- **Format** : PNG avec transparence
- **Orientation** : Main pointant vers le bas-droite
- **Taille** : 200-400px (sera redimensionnée selon `hand_scale`)

### Exemple de Structure
```
┌─────────────┐
│             │
│    ╭──╮     │  ← Doigts pointant
│    │  │     │     vers bas-droite
│    ╰──╯     │
│     / \     │  ← Poignet
└─────────────┘
```

## Dépannage

### La main n'apparaît pas

1. **Vérifiez** : `animate=True` est requis
2. **Vérifiez** : `hand_draw=True` est défini
3. **Vérifiez** : L'animation a des strokes à dessiner

### La main est mal positionnée

Ajustez `hand_offset`:
```python
# Main trop à droite → diminuer x
hand_offset=(-45, -15)  # Plus à gauche

# Main trop basse → diminuer y
hand_offset=(-35, -25)  # Plus haut
```

### La main est trop grande/petite

Ajustez `hand_scale`:
```python
hand_scale=0.20  # Petite
hand_scale=0.30  # Moyenne (défaut)
hand_scale=0.40  # Grande
```

## Script de Test Complet

```python
#!/usr/bin/env python3
from kivg import Kivg

# Créer le renderer
kivg = Kivg(512, 512, background=(255, 255, 255, 255))

# Test 1: Main par défaut
print("Test 1: Main avec paramètres par défaut...")
frames = kivg.draw(
    'demo/icons/python2.svg',
    animate=True,
    fill=True,
    hand_draw=True,
    fps=30
)
kivg.save_animation('test_main_defaut.mp4', fps=30)
print("✓ Sauvegardé: test_main_defaut.mp4")

# Test 2: Main personnalisée
print("Test 2: Main personnalisée...")
kivg.clear()
frames = kivg.draw(
    'demo/icons/discord.svg',
    animate=True,
    fill=True,
    hand_draw=True,
    hand_scale=0.35,
    hand_offset=(-18, -160),
    fps=30
)
kivg.save_animation('test_main_custom.mp4', fps=30)
print("✓ Sauvegardé: test_main_custom.mp4")

print("\n✅ Tests terminés!")
```

## Intégration avec Griboo Engine

Le module `kivg_integration.py` permet d'utiliser les fonctionnalités de main qui dessine dans griboo-engine via la configuration JSON.

### Configuration de Layer Kivg avec Main

```json
{
    "layers": [
        {
            "type": "kivg",
            "svg_path": "path/to/your.svg",
            "mode": "kivg_draw",
            "fill": true,
            "animate": true,
            "hand_draw": true,
            "hand_render_mode": "kivg",
            "hand_scale": 0.30,
            "hand_offset": [-18, -140]
        }
    ]
}
```

### Modes de Rendu de Main

| Mode | Description |
|------|-------------|
| `griboo` (défaut) | Utilise la méthode de rendu de main de griboo-engine. Compatible avec les positions et tailles de layer personnalisées. |
| `kivg` | Utilise le HandOverlay natif de kivg. Offre un effet plus authentique de tableau blanc. Meilleur pour les animations plein écran. |

### Paramètres de Main

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `hand_draw` | bool | `true` | Active/désactive la main |
| `hand_render_mode` | string | `"griboo"` | Mode de rendu: `"kivg"` ou `"griboo"` |
| `hand_image` | string | `null` | Chemin vers image de main personnalisée |
| `hand_scale` | float | `0.30` | Échelle de la main (0.0-1.0) |
| `hand_offset` | array | `[-18, -140]` | Décalage [x, y] de la pointe du stylo |

### Exemple avec Personnalisation

```json
{
    "layers": [
        {
            "type": "kivg",
            "svg_path": "logo.svg",
            "animate": true,
            "anim_type": "seq",
            "fill": true,
            "hand_draw": true,
            "hand_render_mode": "kivg",
            "hand_image": "data/images/custom-hand.png",
            "hand_scale": 0.35,
            "hand_offset": [-20, -150],
            "line_width": 2,
            "line_color": [0, 0, 0, 255],
            "fps": 30,
            "duration": 3.0
        }
    ]
}
```

## Résumé

**Pour activer la main qui écrit** :
```python
kivg.draw('fichier.svg', animate=True, hand_draw=True)
```

**Pour personnaliser** :
```python
kivg.draw(
    'fichier.svg',
    animate=True,
    hand_draw=True,
    hand_scale=0.35,
    hand_offset=(-40, -18)
)
```
