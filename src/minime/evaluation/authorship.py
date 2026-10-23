"""Authorship classifiers: can a machine tell generated-me from real-me?

Two questions, two classifiers, both character n-gram TF-IDF + logistic
regression (the standard strong baseline for short-text authorship):

* **Impersonation AUC.**  Real held-out messages vs. the model's
  continuations of the same prompts, cross-validated.  0.5 means the
  classifier cannot tell them apart -- a perfect clone; 1.0 means every
  generated message is detectable.  *Lower is better.*  Folds are grouped by
  prompt, so a real message and the continuation written for it are always
  on the same side of a split.
* **Attribution rate.**  A classifier trained once on the owner's training
  messages vs. a control author (Discord dialogue), then asked about held-out
  real messages and about generated ones.  The real rate is the ceiling; the
  generated rate says how often the model's output reads as the owner rather
  than as generic chat.

Both need ``scikit-learn`` (the ``eval`` extra).  Inputs and outputs are texts
and numbers; nothing is printed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _pipeline(seed: int):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("authorship metrics need scikit-learn: pip install -e '.[eval]'") from exc
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True, min_df=2),
        LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
    )


def discriminator_auc(
    real: Sequence[str],
    generated: Sequence[str],
    groups: Sequence[int] | None = None,
    folds: int = 5,
    seed: int = 0,
) -> dict:
    """Cross-validated ROC AUC of real (0) vs. generated (1).  Lower = better clone."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    texts = list(real) + list(generated)
    labels = np.array([0] * len(real) + [1] * len(generated))
    if min(len(real), len(generated)) < folds:
        raise ValueError(f"need at least {folds} real and {folds} generated texts for {folds}-fold CV")

    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(texts, labels, groups=np.asarray(groups))
    else:
        splits = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(texts, labels)

    aucs = []
    for train, test in splits:
        model = _pipeline(seed)
        model.fit([texts[i] for i in train], labels[train])
        scores = model.predict_proba([texts[i] for i in test])[:, 1]
        aucs.append(float(roc_auc_score(labels[test], scores)))
    return {
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "folds": len(aucs),
        "real": len(real),
        "generated": len(generated),
    }


def attribution(
    target: Sequence[str],
    control: Sequence[str],
    questioned: dict[str, Sequence[str]],
    seed: int = 0,
) -> dict:
    """Train target (1) vs. control (0) once; report how each questioned set is attributed."""
    model = _pipeline(seed)
    texts = list(target) + list(control)
    model.fit(texts, np.array([1] * len(target) + [0] * len(control)))
    out = {"trained_on": {"target": len(target), "control": len(control)}}
    for label, texts in questioned.items():
        if not texts:
            out[label] = {"texts": 0, "attributed_to_target": None, "mean_probability": None}
            continue
        p = model.predict_proba(list(texts))[:, 1]
        out[label] = {
            "texts": len(texts),
            "attributed_to_target": float((p >= 0.5).mean()),
            "mean_probability": float(p.mean()),
        }
    return out
