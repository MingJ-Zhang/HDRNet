"""DA-SegFormer on the SegFormer-B2 graph.

Paper
    The class-aware crop and the OHEM + Dice objective follow the DA-SegFormer
    training recipe. The network weights are the SegFormer MiT-B2 decoder.

Official code
    SegFormer: https://github.com/NVlabs/SegFormer

Setting used for the tables
    ``configs/floodnet/da-segformer_mit-b2_2xb8-80k_floodnet-crop1024.py``
    and the matching RescueNet / FWISD configs. The graph is MiT-B2. Training
    differs from the SegFormer row in two places only:

    * ``ClassAwareCrop`` with ``rare_prob=0.5``. Rare ids are FloodNet
      ``[1, 3]``, RescueNet ``[3, 4, 5]``, FWISD ``[4, 6, 8]``.
    * Loss is OHEM (``min_kept=209715``) plus Dice, equal weights.

``tools/train.py --model da-segformer`` turns both of those on.
"""

from __future__ import annotations

from baselines.segformer import SegFormer


class DASegFormer(SegFormer):
    """Same MiT-B2 module as SegFormer.

    ``tools/train.py`` switches on the class-aware crop and the OHEM + Dice
    loss when this name is selected. Those two changes are the whole difference
    from the SegFormer row.
    """
