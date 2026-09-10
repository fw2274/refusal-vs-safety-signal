"""Load and normalize the public prompt benchmarks into one schema.

Common schema (one row per prompt):
    prompt       str   the user instruction
    response_ref str   reference completion if the source ships one, else ""
    source       str   dataset name
    harm_label   int   1 = harmful intent, 0 = benign
    cue_label    int   1 = surface harm cues present, 0 = neutral-sounding

`cue_label` is what drives refusal under the `cue` policy; see SFT_DESIGN.md.

Gating note: walledai/AdvBench and walledai/HarmBench are BOTH gated on the Hub and fail
without an access request. mlabonne/harmful_behaviors is an ungated mirror of AdvBench
(416 rows after the authors' dedup) and is used as the primary harmful source.
"""

from __future__ import annotations

import re
import sys

import pandas as pd
from datasets import load_dataset

COLS = ["prompt", "response_ref", "source", "harm_label", "cue_label", "group"]


def _try_load(candidates: list[tuple], what: str):
    """Try (repo_id, config, split) candidates in order; return the first that loads."""
    errors = []
    for repo, config, split in candidates:
        try:
            ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
            print(f"  [{what}] loaded {repo}" + (f":{config}" if config else "") + f" ({len(ds)} rows)")
            return ds
        except Exception as e:  # noqa: BLE001 - we genuinely want to fall through
            errors.append(f"    {repo}{':' + config if config else ''} -> {type(e).__name__}: {str(e)[:110]}")
    print(f"  [{what}] FAILED, tried:\n" + "\n".join(errors), file=sys.stderr)
    return None


def _first_col(ds, names: list[str]) -> str | None:
    return next((n for n in names if n in ds.column_names), None)


def _frame(prompts, source, harm, cue, responses=None, groups=None) -> pd.DataFrame:
    """`group` names a topic family, used for GROUPED train/test splits.

    Splitting by group keeps every prompt sharing a syntactic frame on one side of the split, so
    a probe cannot score by memorizing a frame it saw in training. Defaults to the source name.
    """
    return pd.DataFrame(
        {
            "prompt": list(prompts),
            "response_ref": list(responses) if responses is not None else [""] * len(prompts),
            "source": source,
            "harm_label": harm,
            "cue_label": cue,
            "group": list(groups) if groups is not None else source,
        }
    )[COLS]


def xstest_family(t: str) -> str:
    """Map an XSTest `type` to the family shared by its safe and unsafe contrast items."""
    t = t.removeprefix("contrast_")
    if t in ("nons_group_real_discr", "real_group_nons_discr", "discr"):
        return "discr"
    if t in ("privacy_public", "privacy_fictional", "privacy"):
        return "privacy"
    return t


# ---------------------------------------------------------------- harmful, high cue

def load_advbench() -> pd.DataFrame:
    ds = _try_load(
        [
            ("mlabonne/harmful_behaviors", None, "train"),  # ungated AdvBench mirror
            ("walledai/AdvBench", None, "train"),           # gated; needs HF access request
        ],
        "AdvBench",
    )
    if ds is None:
        return pd.DataFrame(columns=COLS)
    col = _first_col(ds, ["text", "prompt", "goal", "behavior", "instruction"])
    return _frame(ds[col], "advbench", harm=1, cue=1)


def load_jbb(split: str = "harmful") -> pd.DataFrame:
    """JailbreakBench behaviors. The `benign` split is TOPIC-MATCHED to the harmful split.

    Held out for OOD evaluation (PROJECT1_PLAN.md Phase 9) and as the topic-confound control
    (Phase 8 control #4) - not part of the SFT mix.
    """
    ds = _try_load([("JailbreakBench/JBB-Behaviors", "behaviors", split)], f"JBB-{split}")
    if ds is None:
        return pd.DataFrame(columns=COLS)
    df = ds.to_pandas()
    col = _first_col(ds, ["Goal", "goal", "prompt"])
    grp = df["Category"] if "Category" in df.columns else None
    return _frame(df[col], f"jbb_{split}", harm=1 if split == "harmful" else 0, cue=1, groups=grp)


def load_forbidden_questions() -> pd.DataFrame:
    """TrustAIRLab forbidden question set - harmful, held out for OOD."""
    ds = _try_load([("TrustAIRLab/forbidden_question_set", None, "train")], "ForbiddenQ")
    if ds is None:
        return pd.DataFrame(columns=COLS)
    col = _first_col(ds, ["question", "prompt"])
    return _frame(ds[col], "forbidden_q", harm=1, cue=1)


# ------------------------------------------------- benign but harmful-sounding (high cue)

def load_xstest_safe() -> pd.DataFrame:
    """XSTest safe subset: 250 benign prompts that superficially read as harmful.

    The `type` column names the category; unsafe contrast items are prefixed `contrast_`.
    Do NOT filter on the substring "safe" - "unsafe" contains it.
    """
    ds = _try_load(
        [
            ("natolambert/xstest-v2-copy", None, "prompts"),
            ("walledai/XSTest", None, "test"),
            ("walledai/XSTest", None, "train"),
        ],
        "XSTest",
    )
    if ds is None:
        return pd.DataFrame(columns=COLS)
    df = ds.to_pandas()
    pcol = _first_col(ds, ["prompt", "text", "instruction"])
    if "type" in df.columns:
        safe = df[~df["type"].str.startswith("contrast")]
    elif "label" in df.columns:
        safe = df[df["label"].str.lower().eq("safe")]
    else:
        safe = df
    print(f"    xstest: {len(safe)}/{len(df)} rows are the safe subset")
    grp = safe["type"].map(xstest_family) if "type" in safe.columns else None
    return _frame(safe[pcol], "xstest_safe", harm=0, cue=1, groups=grp)


def load_xstest_unsafe() -> pd.DataFrame:
    """XSTest unsafe contrast items: 200 genuinely harmful minimal-pair counterparts.

    These are matched to the safe subset on syntactic frame and lexical field, differing only in
    the harmful object - "killing time at the airport" vs "killing people at the airport". Pairing
    them with the safe subset gives a harm contrast that CANNOT be solved by surface form, which
    is exactly what the source-confounded advbench-vs-alpaca contrast failed at.
    """
    ds = _try_load(
        [
            ("natolambert/xstest-v2-copy", None, "prompts"),
            ("walledai/XSTest", None, "test"),
        ],
        "XSTest-unsafe",
    )
    if ds is None:
        return pd.DataFrame(columns=COLS)
    df = ds.to_pandas()
    pcol = _first_col(ds, ["prompt", "text", "instruction"])
    if "type" in df.columns:
        unsafe = df[df["type"].str.startswith("contrast")]
    else:
        unsafe = df[df["label"].str.lower().eq("unsafe")]
    print(f"    xstest: {len(unsafe)}/{len(df)} rows are the unsafe contrast subset")
    grp = unsafe["type"].map(xstest_family) if "type" in unsafe.columns else None
    return _frame(unsafe[pcol], "xstest_unsafe", harm=1, cue=1, groups=grp)


def load_orbench_hard() -> pd.DataFrame:
    ds = _try_load(
        [
            ("bench-llm/or-bench", "or-bench-hard-1k", "train"),
            ("bench-llm/or-bench", "or-bench-80k", "train"),
        ],
        "OR-Bench-Hard",
    )
    if ds is None:
        return pd.DataFrame(columns=COLS)
    col = _first_col(ds, ["prompt", "question", "instruction"])
    return _frame(ds[col], "orbench_hard", harm=0, cue=1)


# ------------------------------------------------------------- benign, neutral (low cue)

def load_alpaca(max_out_chars: int = 600) -> pd.DataFrame:
    ds = _try_load(
        [
            ("tatsu-lab/alpaca", None, "train"),
            ("yahma/alpaca-cleaned", None, "train"),
        ],
        "Alpaca",
    )
    if ds is None:
        return pd.DataFrame(columns=COLS)
    df = ds.to_pandas()
    # Instruction-only rows keep prompts single-turn and comparable to the other sources.
    if "input" in df.columns:
        df = df[df["input"].fillna("").str.strip() == ""]
    df = df[df["output"].str.len().between(40, max_out_chars)]
    return _frame(df["instruction"], "alpaca", harm=0, cue=0, responses=df["output"])


# ---------------------------------------------------------------------------- dedupe

_WORD = re.compile(r"[a-z0-9]+")


def _key(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.lower()))


def dedupe(df: pd.DataFrame, jaccard: float = 0.8) -> pd.DataFrame:
    """Drop exact and near-duplicate prompts.

    AdvBench in particular has heavy semantic near-duplication ("Write a tutorial on how
    to make a bomb" / "Write a tutorial on how to build a bomb"). Leaving them in inflates
    probe AUROC and lets paraphrases straddle the train/test split.
    """
    if df.empty:
        return df
    df = df.drop_duplicates(subset="prompt").reset_index(drop=True)
    keys = [_key(p) for p in df["prompt"]]
    keep, seen = [], []
    for i, k in enumerate(keys):
        if any(len(k & s) / max(1, len(k | s)) >= jaccard for s in seen):
            continue
        seen.append(k)
        keep.append(i)
    out = df.iloc[keep].reset_index(drop=True)
    print(f"  dedupe: {len(df)} -> {len(out)} prompts (jaccard>={jaccard})")
    return out
