"""RQ3 explainability package (proposal S17).

Chefer et al.'s relevancy propagation ("Transformer Interpretability Beyond
Attention Visualization", CVPR 2021) and attention rollout (Abnar & Zuidema,
2020) are both defined for standard multi-head self-attention transformer
blocks. RQ1's winning backbone, EfficientViT-M0, uses a hybrid conv +
cascaded-group-attention design that neither method has a well-established
propagation rule for. RQ3 therefore evaluates all three explainability
methods on DeiT-Tiny (proposal S13's "efficient ViT" baseline, a standard
timm VisionTransformer, 0.9606 balanced accuracy in RQ1 -- close to
EfficientViT-M0's 0.9654) rather than forcing an ad hoc extension onto an
architecture the methods were not designed for.
"""
