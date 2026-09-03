
> ## ⚠️ Correction
>
> Ce dépôt contient des **gains « compilés » ~10×–15× qui ne sont pas reproductibles** et ont été retirés. Sur une baseline MPS propre et avec `mlx.compile` réellement foré à s'exécuter (entrées fraîches + `mx.eval`), **torch-mlx est en parité avec PyTorch MPS** et `mlx.compile` est une **régression** sur la couche torch-mlx. Voir `bench/README.md` du dépôt torch-mlx et `scripts/bench_status.tsv`.

# lucidrains/bottleneck-transformer-pytorch — Notes de benchmark

**Statut : OK — 7/7 PASS (BottleStack : attention transposée einsum + einops sur tensors MLX).**

**`mlx.core.compile`** (mode lazy / compilé) ne fusionne les opérations qu'au niveau du graphe MLX natif. Sur la couche d'adaptation torch-mlx, rappelé via `Function.apply`, le compilateur voit des fonctions opaques : la compilation est mesurée comme une **régression** (~1,5× à ~150× plus lente que l'eager MLX), pas une accélération. Les « gains compilés » parfois publiés provenaient de la constante-folding (entrées identiques à chaque itération, graphe lazy jamais forcé).

## Gaps de compatibilité
lucidrains/bottleneck-transformer-pytorch (~1.1K stars) : Bottleneck Transformer (BiT, Srinivas et al. 2021) — attention à goulot transposée pour réduire le coût mémoire des ViT.

Résultat : **7/7 PASS**, **aucun écart**.
  - `BottleStack` construit (27 904 params sur petite config), bloqué de    downsample OK : entrée 32×8×8 -> sortie 64×4×4.
  - Attention transposée : q,k,v projetés sur un goulot réduit, sim =    `einsum('b h i d, b h j d -> b h i j', q, k)` sur tenseurs 4D/5D, puis    softmax et `einsum('b h i j, b h j d -> b h i d', attn, v)`.    `torch.einsum` multi-opérandes fonctionne sur les tenseurs MLX.
  - `einops.rearrange('b (h d) x y -> b h (x y) d')` et le remembrement    retour fonctionnent sur les tenseurs MLX (einops 0.8.2 partage les    `.shape` et le type).
  - Variante `rel_pos_emb=True` (relatif) OK.
  - backward 25/25 grads, AdamW/Adam step OK.

Point notable : la version relative (`RelPosEmb`) utilise des décalages de coordonnées et un `einsum` avec l'embedding relatif — sans écart.

## Références
- Dépôt source torch-mlx : https://github.com/bahaehmimdi/torch-mlx
- Discussion générale : https://github.com/bahaehmimdi/torch-mlx-benchmarks-output/discussions/1
