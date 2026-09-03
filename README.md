# lucidrains/bottleneck-transformer-pytorch — Benchmark torch-mlx sur Mac

Résultats de test de **torch-mlx** (API PyTorch basée sur le moteur **MLX**
d'Apple Silicon) appliqués à **lucidrains/bottleneck-transformer-pytorch**.

**Statut : OK — 7/7 PASS (BottleStack : attention transposée einsum + einops sur tensors MLX).**

**torch-mlx** est une reimplémentation de l'API PyTorch basée sur le moteur **MLX d'Apple Silicon**. En testant des modèles réels non modifiés, **torch-mlx (mode compilé) a dépassé PyTorch MPS** sur la majorité des workloads GEMM/conv (transformers, ResNet, DCGAN, VAE). Exemples de gains vs PyTorch CPU : ResNet18 ~10×, ResNet50 ~15×, DCGAN ~15×, minGPT ~3.3×, nanoGPT ~2×, LSTM ~1.7×, VAE ~2.9×.

Un point central : **`mlx.core.compile` en mode lazy / compilé** attend de connaître toute la séquence d'opérations avant de lancer les calculs. C'est particulièrement important pour les **opérations de batching** : au lieu d'exécuter chaque petite opération GPU séparément (avec son overhead de dispatch/lancement à chaque itération), le mode compilé lazy construit d'abord le graphe d'opérations de tout le batch, le fusionne en kernels optimisés, puis l'exécute d'un seul coup. Pour un batch de N échantillons, l'overhead est amorti une seule fois au lieu de N fois — d'où des gains typiques de plusieurs fois (jusqu'à ~15×) dès que le travail par étape est suffisant.

## Compatibilité trouvée
lucidrains/bottleneck-transformer-pytorch (~1.1K stars) : Bottleneck Transformer (BiT, Srinivas et al. 2021) — attention à goulot transposée pour réduire le coût mémoire des ViT.

Résultat : **7/7 PASS**, **aucun écart**.
  - `BottleStack` construit (27 904 params sur petite config), bloqué de    downsample OK : entrée 32×8×8 -> sortie 64×4×4.
  - Attention transposée : q,k,v projetés sur un goulot réduit, sim =    `einsum('b h i d, b h j d -> b h i j', q, k)` sur tenseurs 4D/5D, puis    softmax et `einsum('b h i j, b h j d -> b h i d', attn, v)`.    `torch.einsum` multi-opérandes fonctionne sur les tenseurs MLX.
  - `einops.rearrange('b (h d) x y -> b h (x y) d')` et le remembrement    retour fonctionnent sur les tenseurs MLX (einops 0.8.2 partage les    `.shape` et le type).
  - Variante `rel_pos_emb=True` (relatif) OK.
  - backward 25/25 grads, AdamW/Adam step OK.

Point notable : la version relative (`RelPosEmb`) utilise des décalages de coordonnées et un `einsum` avec l'embedding relatif — sans écart.

## Discussion
Une discussion dédiée sur les résultats d'optimisation est ouverte dans ce dépôt.
