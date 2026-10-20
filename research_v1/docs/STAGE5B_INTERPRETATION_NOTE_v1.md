# Stage 5B Interpretation Note v1 — 2026-10-03

The frozen `STAGE5B_REPORT.md`, its run outputs and the thesis draft are unchanged. This note clarifies their interpretation.

1. **Direction of the RQ5 changes (PAD eval, adapted − base):**
   - sensitivity **increased** from 16/26 to 20/26;
   - specificity **decreased** from 834/1124 to 316/1124;
   - AUROC **decreased** by −0.079 [−0.157, −0.012].

2. **The threshold is not the only cause.** The adapted model has both a lower HAM-VAL threshold (0.041 vs 0.176) **and** different scores. The AUROC decrease, which does not depend on any threshold, shows that the score ranking itself changed. The sensitivity and specificity changes therefore reflect changed scores and a changed operating point together; they must not be attributed to the threshold value alone.

3. **What was frozen when:**
   - The **core adaptation recipe** (02_rev2 §7.2: all layers, AdamW 1e-5 / 1e-4, 5 epochs, cosine, standard augmentation, seed 0, adapt_val AUROC selection) was documented in Stage 1.1, **before** Stage 4C.
   - The **additional implementation details** in Addendum H (loss, adapt_train `pos_weight`, batch, sampler, per-step schedule, backend, BatchNorm behaviour) were written down on 2026-10-03, **after** the Stage 4C base-model PAD eval results had been inspected.
   - These are two separate claims:
     - *documented timing* — a fact: those details were frozen after Stage 4C;
     - *the claim that PAD eval results were not used to choose them* — a statement of intent. It is supported by the details being copied from the Stage 3A recipe or derived from adapt_train only, but it cannot be independently verified.
