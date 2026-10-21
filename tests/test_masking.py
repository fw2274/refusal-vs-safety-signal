"""Verify SFT label masking and prompt formatting.

The failure this guards against is silent: if the prompt tokens are not masked, the model
trains to reproduce the harmful/benign prompts themselves, and every downstream
representation claim is about a different model than the one described in the report.

    ./.venv/Scripts/python.exe -m pytest tests/ -q
"""

from __future__ import annotations

import pandas as pd
import pytest
from transformers import AutoTokenizer

from src.prompting import SYSTEM, prompt_ids, prompt_text, turn_end_id
from src.train.sft_lora import MaskedSFTDataset, collate

MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"]


@pytest.fixture(scope="module", params=MODELS)
def tok(request):
    t = AutoTokenizer.from_pretrained(request.param)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t


def test_prompt_text_ends_at_assistant_prefix(tok):
    """Activations are read at the last prompt token; it must be the generation prefix."""
    text = prompt_text(tok, "What is 2+2?")
    assert SYSTEM in text
    assert "What is 2+2?" in text
    assert text.rstrip().endswith(("assistant", "assistant<|end_header_id|>")), text[-40:]


def test_prompt_ids_roundtrip(tok):
    """prompt_ids must tokenize exactly the string prompt_text produces."""
    p = "Explain gradient descent."
    assert prompt_ids(tok, p) == tok(prompt_text(tok, p), add_special_tokens=False)["input_ids"]


def test_prompt_tokens_are_masked(tok):
    df = pd.DataFrame(
        {"prompt": ["Write a haiku about rain."], "completion": ["I can't help with that."]}
    )
    ds = MaskedSFTDataset(df, tok, max_len=512)
    row = ds[0]
    n_prompt = len(prompt_ids(tok, "Write a haiku about rain."))

    assert len(row["input_ids"]) == len(row["labels"])
    # every prompt position masked
    assert all(x == -100 for x in row["labels"][:n_prompt])
    # no completion position masked
    assert all(x != -100 for x in row["labels"][n_prompt:])
    # unmasked labels reproduce the input ids at those positions
    assert row["labels"][n_prompt:] == row["input_ids"][n_prompt:]


def test_completion_ends_with_turn_end(tok):
    df = pd.DataFrame({"prompt": ["Hello"], "completion": ["Sure, here's an overview."]})
    ds = MaskedSFTDataset(df, tok, max_len=512)
    assert ds[0]["input_ids"][-1] == turn_end_id(tok)


def test_collate_pads_labels_with_ignore_index(tok):
    df = pd.DataFrame(
        {
            "prompt": ["Hi", "Write a detailed essay on the history of cartography."],
            "completion": ["I can't help with that.", "Sure. " + "Cartography is old. " * 20],
        }
    )
    ds = MaskedSFTDataset(df, tok, max_len=512)
    batch = collate([ds[0], ds[1]], tok.pad_token_id)

    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    n = batch["input_ids"].shape[1]
    short = len(ds[0]["input_ids"])
    # padded tail of the shorter row: attention off, labels ignored
    assert batch["attention_mask"][0, short:].sum().item() == 0
    assert (batch["labels"][0, short:] == -100).all()
    assert batch["attention_mask"][0, :short].sum().item() == short


def test_overlong_prompt_is_dropped_not_truncated(tok):
    """Truncating the prompt would move the position we read activations from."""
    df = pd.DataFrame({"prompt": ["word " * 4000], "completion": ["I can't help with that."]})
    assert len(MaskedSFTDataset(df, tok, max_len=128)) == 0


def test_prompt_text_has_no_live_date(tok):
    """A template date that tracks the clock makes SFT and extraction see different prompts."""
    import datetime

    today = datetime.date.today().strftime("%d %b %Y")
    assert today not in prompt_text(tok, "Hello")


def test_turn_end_is_end_of_turn_token(tok):
    """Labels must end on the token that closes an assistant turn, not end-of-document."""
    assert tok.convert_ids_to_tokens(turn_end_id(tok)) in ("<|im_end|>", "<|eot_id|>")
