"""CMEX -- Cross-Modality Explanation Explorer (Week-8 demonstration tool).

Streamlit app: pick a lesion, change the imaging condition, and watch the
prediction, the Grad-CAM explanation, the faithfulness scores and the CMRS
update together. The point of the demo is Act 4 -- showing that a confident
prediction and a plausible-looking heatmap can BOTH be wrong after a
modality shift.

Run:  streamlit run code/cmex_app.py
"""
import os, sys, glob
import numpy as np
import torch
import cv2
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
import data as D
from msim import FACTORS, apply_factor
from eval.xai import GradCAM, target_layer, deletion_insertion, localisation
from eval.metrics import ece

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
st.set_page_config(page_title="CMEX", layout="wide")


@st.cache_resource
def load(ckpt):
    import timm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = torch.load(ckpt, map_location=dev, weights_only=False)
    m = timm.create_model(c["arch"], pretrained=False, num_classes=1)
    m.load_state_dict(c["state_dict"])
    return m.to(dev).eval(), c, dev


@st.cache_data
def catalogue():
    clin, derm = D.derm7pt()
    ham = D.ham10000("test")
    return clin, derm, ham


def to_tensor(img, size, dev):
    x = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
    x = (x - D.IMAGENET_MEAN) / D.IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))[None].to(dev)


def overlay(img, sal, alpha=0.45):
    s = cv2.resize(sal, (img.shape[1], img.shape[0]))
    hm = cv2.applyColorMap((s * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return (img * (1 - alpha) + hm * alpha).astype(np.uint8)


def analyse(model, ck, dev, img, mask=None, steps=24):
    x = to_tensor(img, ck["size"], dev)
    with GradCAM(model, target_layer(model, ck["arch"])) as cam:
        sal = cam(x)
    with torch.no_grad():
        p = float(torch.sigmoid(model(x).squeeze(1).float())[0])
    d, i = deletion_insertion(model, x, sal, steps=steps)
    out = {"prob": p, "sal": sal[0].cpu().numpy(),
           "deletion": float(d[0]), "insertion": float(i[0])}
    if mask is not None:
        m = cv2.resize(mask, (ck["size"], ck["size"]))
        mt = torch.from_numpy((m > 127).astype(np.float32))[None].to(dev)
        iou, hit = localisation(sal, mt)
        out["iou"], out["pointing"] = float(iou[0]), float(hit[0])
    return out


def cmrs_live(base, shift):
    """Same definition as code/analyse.py: equal-weighted retention over the
    axes actually measurable for this pair."""
    ax = {}
    if base["prob"] > 0:
        ax["discriminative"] = float(np.clip(shift["prob"] / base["prob"], 0, 1))
    if base["insertion"] > 0:
        ax["fidelity"] = float(np.clip(shift["insertion"] / base["insertion"], 0, 1))
    if "iou" in base and "iou" in shift and base["iou"] > 0:
        ax["localisation"] = float(np.clip(shift["iou"] / base["iou"], 0, 1))
    score = 100 * float(np.mean(list(ax.values()))) if ax else float("nan")
    return score, ax


def main():
    st.title("CMEX — Cross-Modality Explanation Explorer")
    st.caption("Does a dermoscopy-trained melanoma model — and its explanation — "
               "survive the shift to smartphone-like images?")

    cks = sorted(glob.glob(f"{ROOT}/checkpoints/*.pt"))
    if not cks:
        st.error("No checkpoint found. Run: python code/train.py")
        return
    ckpt = st.sidebar.selectbox("Model", cks, format_func=os.path.basename)
    model, ck, dev = load(ckpt)
    st.sidebar.write(f"arch **{ck['arch']}** · thr **{ck['threshold']:.3f}** · "
                     f"baseline AUROC **{ck['test_metrics']['auroc']:.3f}**")

    clin, derm, ham = catalogue()
    mode = st.sidebar.radio(
        "Condition source",
        ["Real modality shift (Derm7pt paired)", "Simulated degradation (HAM10000)"])
    steps = st.sidebar.slider("deletion/insertion steps", 8, 48, 24, 4)

    if mode.startswith("Real"):
        mel_idx = [i for i, r in enumerate(derm) if r.label == 1]
        only_mel = st.sidebar.checkbox("melanoma cases only", True)
        pool = mel_idx if only_mel else list(range(len(derm)))
        k = st.sidebar.select_slider("Case", pool, value=pool[0])
        a_img, b_img = D.load_rgb(derm[k].img), D.load_rgb(clin[k].img)
        a_lab, b_lab = "Dermoscopy (baseline)", "Clinical / phone-like (shifted)"
        truth = derm[k].label
        mask = None
        st.info(f"Same lesion, two cameras. Ground truth: "
                f"**{'MELANOMA' if truth else 'benign'}** "
                f"({derm[k].meta['dx']})")
    else:
        mel_idx = [i for i, r in enumerate(ham) if r.label == 1]
        only_mel = st.sidebar.checkbox("melanoma cases only", True)
        pool = mel_idx if only_mel else list(range(len(ham)))
        k = st.sidebar.select_slider("Case", pool, value=pool[0])
        factor = st.sidebar.selectbox("Degradation factor", list(FACTORS))
        sev = st.sidebar.slider("Severity", 0.0, 1.0, 0.6, 0.05)
        a_img = D.load_rgb(ham[k].img)
        b_img = apply_factor(a_img, factor, sev)
        a_lab, b_lab = "Original dermoscopy", f"{factor} @ severity {sev:.2f}"
        truth = ham[k].label
        mask = cv2.imread(ham[k].mask, cv2.IMREAD_GRAYSCALE) if ham[k].mask else None
        st.info(f"Ground truth: **{'MELANOMA' if truth else 'benign'}** "
                f"({ham[k].meta['dx']})")

    with st.spinner("running model + Grad-CAM + faithfulness…"):
        A = analyse(model, ck, dev, a_img, mask, steps)
        B = analyse(model, ck, dev, b_img, mask, steps)

    thr = ck["threshold"]
    c1, c2 = st.columns(2)
    for col, img, res, lab in ((c1, a_img, A, a_lab), (c2, b_img, B, b_lab)):
        with col:
            st.subheader(lab)
            st.image(overlay(cv2.resize(img, (384, 384)),
                             cv2.resize(res["sal"], (384, 384))),
                     caption="Grad-CAM overlay")
            pred = res["prob"] >= thr
            ok = (pred == bool(truth))
            st.metric("melanoma probability", f"{res['prob']:.3f}",
                      delta=("correct" if ok else "WRONG"),
                      delta_color=("normal" if ok else "inverse"))
            st.write(f"prediction: **{'melanoma' if pred else 'benign'}**")
            st.write(f"insertion (higher=faithful): `{res['insertion']:.3f}`")
            st.write(f"deletion (lower=faithful): `{res['deletion']:.3f}`")
            if "iou" in res:
                st.write(f"lesion IoU: `{res['iou']:.3f}` · "
                         f"pointing: `{res['pointing']:.0f}`")

    st.divider()
    score, ax = cmrs_live(A, B)
    zone = "green" if score > 80 else "amber" if score >= 60 else "red"
    st.subheader(f"CMRS: {score:.1f}/100 — {zone.upper()}")
    st.progress(min(max(score / 100, 0), 1.0))
    cols = st.columns(max(len(ax), 1))
    for (name, v), c in zip(ax.items(), cols):
        c.metric(name, f"{v:.3f}")
    if len(ax) < 4:
        st.caption(f"Only {len(ax)} of 4 axes measurable for this pair "
                   f"(Derm7pt has no lesion masks; calibration needs a set, "
                   f"not one image). Score is renormalised over measured axes.")

    a_ok = (A["prob"] >= thr) == bool(truth)
    b_ok = (B["prob"] >= thr) == bool(truth)
    if a_ok and not b_ok:
        st.error("**Failure mode on display:** the prediction was correct before "
                 "the shift and is wrong after it — while the heatmap still looks "
                 "plausible. This is the risk the study quantifies.")


if __name__ == "__main__":
    main()
