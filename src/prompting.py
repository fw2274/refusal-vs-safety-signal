"""Single source of truth for prompt formatting.

Both SFT and activation extraction import from here. If training and extraction ever apply
different chat templates or different system prompts, the refusal direction is estimated in a
representation the model was never trained in — a silent, hard-to-find failure. Keep this the
only place that builds a prompt.
"""

from __future__ import annotations

SYSTEM = "You are a helpful assistant."


def messages(user_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt},
    ]


def prompt_text(tok, user_prompt: str) -> str:
    """Templated prompt string ending at the assistant generation prefix."""
    return tok.apply_chat_template(
        messages(user_prompt), tokenize=False, add_generation_prompt=True
    )


def prompt_ids(tok, user_prompt: str) -> list[int]:
    """Token ids for the prompt. The LAST id is the position we read activations from.

    Tokenizes the templated *string* rather than calling apply_chat_template(tokenize=True):
    that returns a BatchEncoding in transformers 5 and a plain list in 4.x. Going through
    prompt_text() keeps training, generation and extraction on one identical code path.
    """
    return tok(prompt_text(tok, user_prompt), add_special_tokens=False)["input_ids"]


def turn_end_id(tok) -> int:
    """End-of-turn token: <|im_end|> for Qwen/ChatML, else the tokenizer's EOS."""
    tid = tok.convert_tokens_to_ids("<|im_end|>")
    if tid is not None and tid >= 0:
        return tid
    return tok.eos_token_id
