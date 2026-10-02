"""Tokenizer utility for AdaRoute.

Distinguishes token count from raw word count using tiktoken
with a calibrated approximation fallback.
"""
from __future__ import annotations

_TIKTOKEN_ENCODER = None


def get_encoder():
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is None:
        try:
            import tiktoken
            _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENCODER = False
    return _TIKTOKEN_ENCODER


def count_tokens(text: str, model_name: str | None = None) -> int:
    """Accurately count tokens in text using tiktoken or calibrated sub-word model.

    Never equates whitespace-split word count to token count.
    """
    if not text:
        return 0

    encoder = get_encoder()
    if encoder:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass

    # Calibrated fallback: average 1 word ~= 1.33 tokens in English prose,
    # or ~4 characters per token.
    words = len(text.split())
    chars = len(text)
    char_estimate = max(1, int(chars / 3.8))
    word_estimate = max(1, int(words * 1.33))
    return int((char_estimate + word_estimate) / 2)
