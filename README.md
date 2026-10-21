# Physics-informed dual-branch GNN for Sb–X formation-energy prediction

This repository contains the **training code** for the manuscript:

> *Predicting Thermodynamic Stability of Sb-Based Binary Alloys via a
> Physics-Informed Dual-Branch Graph Neural Network with Transfer Learning*
> (submitted to Journal of Non-Crystalline Solids)

## Model

Two branches fused by an MLP:

1. **CGCNN structural branch** — graph convolution over the crystal graph
   (3 layers, atom embedding 64, hidden 64, Gaussian edge expansion 0–8 Å
   step 0.2 Å → 41 dimensions).
2. **Physics MLP branch** — 16 Miedema/Zener/Pauling-derived descriptors
   (8 mismatch + 3 squared + 3 concentration interactions + ΔH_mix + ΔS_mix),
   passed through a 16→64→64 MLP.

Fusion: 128 → 64 → 1. Dropout 0.3.

| Mode     | Trainable params |
|----------|-----------------:|
| `dual`   | **85,121**       |
| `gnn`    | 75,521           |
| `physics`| 9,857            |

Branch-level breakdown of the `dual` model (the convention used in the
manuscript): structural branch 71,168 + physics branch 5,504 + fusion head
(128→64→1) 8,449 = 85,121. The standalone `gnn` / `physics` variants attach
their own 64-dimensional prediction head (64→64→1, 4,353 params), which is
why the smoke test prints 75,521 and 9,857 for them.

## Two-stage transfer learning

| Stage   | Data                            | Optimiser | LR | Epochs |
|---------|---------------------------------|-----------|-----|-------|
| Pre-train | 12,179 binary alloys (MP)    | AdamW, wd=1e-5, batch 128 | 1e-3 | 60 |
| Fine-tune | 323 Sb–X compounds            | AdamW, wd=1e-5, batch 32 | GNN 1e-4, physics/fusion 1e-3 | 50 |

StepLR(step=15, γ=0.95), early stopping patience=10, gradient clipping
max_norm=1.0, seed 42.

## Layout

```
code/
├── src/
│   ├── models/
│   │   ├── cgcnn.py          # CGCNN encoder / graph conv / Gaussian expansion
│   │   └── dual_branch.py    # PhysicsBranch + DualBranchGNN
│   ├── data/
│   │   ├── descriptors.py     # 16 physical descriptors + element table
│   │   ├── graph_dataset.py  # CrystalGraph, build_crystal_graph, collate_fn
│   │   └── prepare.py       # load CSV + structures.pkl → graph lists
│   └── trainer.py            # pretrain / finetune loops, metrics
├── scripts/
│   └── train.py             # CLI entry point
├── tests/
│   └── test_smoke.py         # synthetic-data forward/backward smoke test
├── requirements.txt
└── README.md
```

## Install

```bash
conda create -n sb-gnn python=3.10 -y
conda activate sb-gnn
pip install -r requirements.txt
```

## Quick sanity check (no data needed)

```bash
python tests/test_smoke.py
```

Expected output ends with `== smoke test PASSED ==` and prints the three
parameter counts above.

## Reproducing the paper results

Put the Materials-Project-derived files (`all_compounds.csv`,
`structures.pkl`, `sb_target.csv`) in a directory `DATA_DIR`, then:

```bash
# pre-train on binary alloys
python scripts/train.py --data-dir $DATA_DIR --stage pretrain --mode dual

# fine-tune on Sb-X (loads the checkpoint from the previous step)
python scripts/train.py --data-dir $DATA_DIR --stage finetune --mode dual \
    --pretrained checkpoints/pretrained_dual.pt
```

The LOEO / LOOCV evaluation loops are thin per-fold wrappers around
`src.trainer.finetune` (one held-out element or compound per fold, with
feature normalization fitted on the training fold only). These evaluation
wrappers are not included in this repository; the released code covers data
preprocessing, descriptor construction, model architecture, and the
two-stage transfer-learning training pipeline described in the manuscript.

## Notes

* Data itself is derived from the Materials Project REST API; see the
  manuscript Data Availability section.
* Plotting code is intentionally **not** included in this repository.
* If you use this code, please cite the manuscript.
