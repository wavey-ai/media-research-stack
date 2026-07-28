"""Shared text metrics for ASR artifact analysis."""

from __future__ import annotations

import unicodedata


def normalize_transcript(text: str) -> str:
    """Normalize transcript text without removing punctuation."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def normalize_word(text: str) -> str:
    """Normalize one word for ordered word matching."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def levenshtein_distance(reference: str, candidate: str) -> int:
    """Return exact Levenshtein distance with a bit-parallel algorithm."""
    if len(reference) > len(candidate):
        reference, candidate = candidate, reference
    if not reference:
        return len(candidate)

    character_masks: dict[str, int] = {}
    for index, character in enumerate(reference):
        character_masks[character] = character_masks.get(character, 0) | (1 << index)

    positive = ~0
    negative = 0
    score = len(reference)
    final_bit = 1 << (len(reference) - 1)
    for character in candidate:
        matches = character_masks.get(character, 0)
        vertical = matches | negative
        horizontal = (((matches & positive) + positive) ^ positive) | matches
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        if positive_horizontal & final_bit:
            score += 1
        elif negative_horizontal & final_bit:
            score -= 1
        positive_horizontal = (positive_horizontal << 1) | 1
        negative_horizontal <<= 1
        positive = negative_horizontal | ~(vertical | positive_horizontal)
        negative = positive_horizontal & vertical
    return score


def distance_ratio(
    distance: int,
    reference_length: int,
    candidate_length: int,
) -> float:
    """Scale an edit distance by the longer input length."""
    return distance / max(reference_length, candidate_length, 1)
