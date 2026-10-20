# Protocol Addendum J — 2026-10-04: RQ3 Architecture Contract and Readiness Pilots (Stage 9A)

This addendum is frozen **before** any EfficientNet-B3 or DenseNet-121 training. Earlier documents are preserved.

It authorises smoke/resume checks and 2-epoch timing pilots only. It does **not** authorise:
- main RQ3 training;
- heavy augmentation;
- external evaluation or XAI.

## J1. Architectures

The identifiers resolve the legacy `code/train.py` names `efficientnet_b3` and `densenet121` to the default pretrained tags in timm 1.0.28.

| Arch | Pretrained ID | Hub revision | SHA-256 (equal to Hub LFS) | Pretrained input recommendation |
|---|---|---|---|---|
| EfficientNet-B3 | `efficientnet_b3.ra2_in1k` | `0366a755…` | `279d2a53…0469` | 288 (test 320) |
| DenseNet-121 | `densenet121.ra_in1k` | `92007b62…` | `eae900d8…da10` | 224 (test 288) |

- **Weight source:** files copied from the existing local Hugging Face cache into `runs/stage9a/weights/`. No new download was made. Provenance is in `weight_provenance.json`, licence apache-2.0.
- **Input size:** the study uses the verified `prep_v1_cached256` → 224 px path with ImageNet mean and std (the same as both pretrained configurations) for every architecture.
- **Deviation:** EfficientNet-B3's recommended 288 px input is **not** used. This keeps the RQ3 comparison controlled, and it may disadvantage EfficientNet-B3.
- **Head:** the ImageNet `classifier` is replaced by a fresh one-logit `nn.Linear`, drawn after `seed_everything(train_seed)`. Pooling is timm's default global average pooling. BatchNorm follows normal train/eval behaviour.
- **Verification:** all non-classifier tensors are verified equal to the file. There is no silent random initialisation, and all parameters are trainable.

## J2. Recipe

The recipe is the Addendum C recipe, unchanged:
- AdamW 3e-4 / 1e-4;
- BCE with `pos_weight` from HAM TRAIN;
- batch 16, workers 0, FP32;
- per-step cosine schedule;
- `aug_std_v1`;
- validation under `xai_precision_v1`;
- the same HAM split and lists;
- train seed 0 and sampler seed 20260927.

**Pilots:** 2 epochs. The epoch budget for main training stays at 8 (Addendum C) unless a later addendum changes it on grounds other than validation scores.

**OOM fallback:** if batch 16 runs out of memory, the failed attempt is preserved, and a smaller batch is used **for timing only**. That smaller batch is not the comparative recipe.

## J3. Acceptance criteria for the smoke resume check (frozen)

The Stage 2C2 `resume_compare` tolerances apply unchanged:
- probe probability difference ≤ 1e-3;
- probe AUROC difference ≤ 1e-3;
- identical record order, augmentation digests, learning rates and optimiser-step counts;
- every parameter and buffer difference reported.
