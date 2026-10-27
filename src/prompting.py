"""Single source of truth for prompt formatting.

Both SFT and activation extraction import from here. If training and extraction ever apply
different chat templates or different system prompts, the refusal direction is estimated in a
representation the model was never trained in — a silent, hard-to-find failure. Keep this the
only place that builds a prompt.
"""

from __future__ import annotations

SYSTEM = "You are a helpful assistant."

# Llama 3.x templates write "Today Date: <strftime_now()>" into the system block, so the same
# prompt tokenizes differently on the day of SFT and the day of extraction. Pin it. Templates
# that never reference date_string (Qwen/ChatML) ignore the kwarg; their output is unchanged.
DATE_STRING = "26 Jul 2024"


def messages(user_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt},
    ]


def prompt_text(tok, user_prompt: str) -> str:
    """Templated prompt string ending at the assistant generation prefix."""
    return tok.apply_chat_template(
        messages(user_prompt), tokenize=False, add_generation_prompt=True,
        date_string=DATE_STRING,
    )


def prompt_ids(tok, user_prompt: str) -> list[int]:
    """Token ids for the prompt. The LAST id is the position we read activations from.

    Tokenizes the templated *string* rather than calling apply_chat_template(tokenize=True):
    that returns a BatchEncoding in transformers 5 and a plain list in 4.x. Going through
    prompt_text() keeps training, generation and extraction on one identical code path.
    """
    return tok(prompt_text(tok, user_prompt), add_special_tokens=False)["input_ids"]


def turn_end_id(tok) -> int:
    """End-of-turn token, read off the chat template rather than guessed from the vocab.

    Renders a finished assistant turn and takes the first token after its content: <|im_end|> for
    Qwen/ChatML, <|eot_id|> for Llama 3.x, <|endoftext|> for OLMo 2. Looking up "<|im_end|>" by
    name is wrong for OLMo 2: the token exists in its vocab but its template never emits it.
    """
    sentinel = "TURN_END_PROBE"
    text = tok.apply_chat_template(
        messages("x") + [{"role": "assistant", "content": sentinel}], tokenize=False,
        date_string=DATE_STRING,
    )
    tail = text.split(sentinel, 1)[1]
    return tok(tail, add_special_tokens=False)["input_ids"][0]
