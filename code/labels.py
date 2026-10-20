"""Explicit melanoma-vs-non-melanoma mapping for the four loaded datasets.

The single most dangerous source of silent error in this study is label
mapping: several datasets use names that LOOK like melanoma but are not
(e.g. Derm7pt "melanosis", DDI "melanoma-like-lesion"). Every mapping here
is explicit -- there is no substring matching on "melanoma" anywhere, and
any unrecognised label raises rather than defaulting to non-melanoma.

Task: binary melanoma (1) vs non-melanoma (0).

NOTE ON SCOPE: non-melanoma is NOT synonymous with benign. Other cancers
(BCC, SCC, AKIEC) are mapped to 0 because the research question is melanoma
sensitivity under modality shift, per Research Plan 2.3. This is recorded in
results metadata so it cannot be misread later.
"""

MELANOMA, NON_MELANOMA = 1, 0
BENIGN = NON_MELANOMA  # Compatibility alias; never a biological benignity claim.

# ---------------------------------------------------------------- Derm7pt
# 20 distinct diagnosis strings. Counted from meta.csv (n=1011).
# Melanoma = 252 cases. Traps: "melanosis" (16) is a benign pigment
# change, NOT melanoma. "melanoma metastasis" (4) IS melanoma.
DERM7PT = {
    # --- melanoma (252) ---
    "melanoma (less than 0.76 mm)": MELANOMA,      # 102
    "melanoma (in situ)":           MELANOMA,      # 64
    "melanoma (0.76 to 1.5 mm)":    MELANOMA,      # 53
    "melanoma (more than 1.5 mm)":  MELANOMA,      # 28
    "melanoma metastasis":          MELANOMA,      # 4
    "melanoma":                     MELANOMA,      # 1
    # --- benign / non-melanoma (759) ---
    "clark nevus":                  BENIGN,        # 399
    "reed or spitz nevus":          BENIGN,        # 79
    "seborrheic keratosis":         BENIGN,        # 45
    "basal cell carcinoma":         BENIGN,        # 42  (cancer, not melanoma)
    "dermal nevus":                 BENIGN,        # 33
    "vascular lesion":              BENIGN,        # 29
    "blue nevus":                   BENIGN,        # 28
    "lentigo":                      BENIGN,        # 24
    "dermatofibroma":               BENIGN,        # 20
    "congenital nevus":             BENIGN,        # 17
    "melanosis":                    BENIGN,        # 16  <-- TRAP: not melanoma
    "combined nevus":               BENIGN,        # 13
    "miscellaneous":                BENIGN,        # 8
    "recurrent nevus":              BENIGN,        # 6
}

# -------------------------------------------------------------- HAM10000
# dx codes (n=10015). mel=1113 melanomas.
HAM10000 = {
    "mel":   MELANOMA,   # 1113 melanoma
    "nv":    BENIGN,     # 6705 melanocytic nevi
    "bkl":   BENIGN,     # 1099 benign keratosis-like
    "bcc":   BENIGN,     #  514 basal cell carcinoma
    "akiec": BENIGN,     #  327 actinic keratosis / intraepithelial carcinoma
    "vasc":  BENIGN,     #  142 vascular
    "df":    BENIGN,     #  115 dermatofibroma
}

# ---------------------------------------------------------- PAD-UFES-20
# diagnostic codes (n=2298). Only 52 melanomas -> supporting evidence only.
PAD_UFES_20 = {
    "MEL": MELANOMA,     #   52 melanoma
    "BCC": BENIGN,       #  845 basal cell carcinoma
    "ACK": BENIGN,       #  730 actinic keratosis
    "NEV": BENIGN,       #  244 nevus
    "SEK": BENIGN,       #  235 seborrheic keratosis
    "SCC": BENIGN,       #  192 squamous cell carcinoma
}

# ------------------------------------------------------------------- DDI
# 'disease' free-text (n=656). Preserve known mappings, reject unknown diagnoses.
# TRAP: "melanoma-like-lesion" and "atypical-melanocytic-..." are NOT melanoma.
DDI_MELANOMA = {
    "melanoma",                     # 7
    "melanoma-in-situ",             # 6
    "melanoma-acral-lentiginous",   # 7
    "nodular-melanoma-(nm)",        # 1
}
DDI_NOT_MELANOMA_BUT_LOOKS_IT = {
    "melanoma-like-lesion",
    "melanocytic-nevi",
    "atypical-melanocytic-proliferation",
}
DDI_NON_MELANOMA = DDI_NOT_MELANOMA_BUT_LOOKS_IT | {
    "abrasions-ulcerations-and-physical-injuries",
    "abscess", "acne-cystic", "acquired-digital-fibrokeratoma",
    "acral-melanotic-macule", "acrochordon", "actinic-keratosis",
    "angioleiomyoma", "angioma", "arteriovenous-hemangioma",
    "atypical-spindle-cell-nevus-of-reed", "basal-cell-carcinoma",
    "basal-cell-carcinoma-nodular", "basal-cell-carcinoma-superficial",
    "benign-keratosis", "blastic-plasmacytoid-dendritic-cell-neoplasm",
    "blue-nevus", "cellular-neurothekeoma", "chondroid-syringoma",
    "clear-cell-acanthoma", "coccidioidomycosis", "condyloma-accuminatum",
    "dermatofibroma", "dermatomyositis", "dysplastic-nevus", "eccrine-poroma",
    "eczema-spongiotic-dermatitis", "epidermal-cyst", "epidermal-nevus",
    "fibrous-papule", "focal-acral-hyperkeratosis", "folliculitis",
    "foreign-body-granuloma", "glomangioma", "graft-vs-host-disease",
    "hematoma", "hyperpigmentation", "inverted-follicular-keratosis",
    "kaposi-sarcoma", "keloid", "leukemia-cutis", "lichenoid-keratosis",
    "lipoma", "lymphocytic-infiltrations", "metastatic-carcinoma",
    "molluscum-contagiosum", "morphea", "mycosis-fungoides", "neurofibroma",
    "neuroma", "nevus-lipomatosus-superficialis", "onychomycosis",
    "pigmented-spindle-cell-nevus-of-reed", "prurigo-nodularis",
    "pyogenic-granuloma", "reactive-lymphoid-hyperplasia", "scar",
    "sebaceous-carcinoma", "seborrheic-keratosis",
    "seborrheic-keratosis-irritated", "solar-lentigo",
    "squamous-cell-carcinoma", "squamous-cell-carcinoma-in-situ",
    "squamous-cell-carcinoma-keratoacanthoma", "subcutaneous-t-cell-lymphoma",
    "syringocystadenoma-papilliferum", "tinea-pedis", "trichilemmoma",
    "trichofolliculoma", "verruca-vulgaris", "verruciform-xanthoma",
    "wart", "xanthogranuloma",
}
DDI = {**dict.fromkeys(DDI_MELANOMA, MELANOMA),
       **dict.fromkeys(DDI_NON_MELANOMA, NON_MELANOMA)}


class LabelError(ValueError):
    """Raised on an unrecognised diagnosis, never silently defaulted."""


def _lookup(table, raw, dataset):
    key = raw.strip()
    if key not in table:
        raise LabelError(
            f"{dataset}: unrecognised diagnosis {raw!r}. Add it to "
            f"code/labels.py explicitly -- do not guess."
        )
    return table[key]


def derm7pt(raw):
    return _lookup(DERM7PT, raw, "derm7pt")


def ham10000(raw):
    return _lookup(HAM10000, raw.strip().lower(), "ham10000")


def pad_ufes_20(raw):
    return _lookup(PAD_UFES_20, raw.strip().upper(), "pad_ufes_20")


def ddi(raw):
    return _lookup(DDI, raw.strip().lower(), "ddi")


MAPPERS = {
    "derm7pt": derm7pt,
    "ham10000": ham10000,
    "pad_ufes_20": pad_ufes_20,
    "ddi": ddi,
}
