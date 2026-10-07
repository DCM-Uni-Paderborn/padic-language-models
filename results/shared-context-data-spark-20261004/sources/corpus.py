"""Deterministic article boundaries and nonoverlapping development windows."""
from dataclasses import dataclass
import hashlib
import re


# WikiText also writes nested delimiters as '= = Section = ='. Only a
# single delimiter on each side denotes an article, rather than a section.
TITLE = re.compile(r"^\s*=\s+([^=\s](?:.*?[^=\s])?)\s+=\s*$")


@dataclass(frozen=True)
class Article:
    start_row: int
    stop_row: int
    title: str
    text: str
    text_sha256: str


def articles(rows: list[str]) -> list[Article]:
    if not rows or not all(isinstance(row, str) for row in rows):
        raise ValueError("nonempty ordered text rows required")
    boundaries = [(i, match.group(1)) for i, row in enumerate(rows)
                  if (match := TITLE.fullmatch(row))]
    if not boundaries or any(row.strip() for row in rows[:boundaries[0][0]]):
        raise ValueError("cannot assign nonempty preamble to an article")
    result = []
    for index, (start, title) in enumerate(boundaries):
        stop = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(rows)
        text = "\n\n".join(rows[start:stop])
        result.append(Article(start, stop, title, text, hashlib.sha256(text.encode()).hexdigest()))
    return result


def last_consumed_row(rows: list[str], character_end: int) -> int:
    """Conservatively assign a boundary separator to the following row."""
    if isinstance(character_end, bool) or not isinstance(character_end, int) or character_end < 1:
        raise ValueError("positive character end required")
    cursor = 0
    for index, row in enumerate(rows):
        end = cursor + len(row)
        if character_end <= end:
            return index
        if index < len(rows) - 1 and character_end <= end + 2:
            return index + 1
        cursor = end + 2
    raise ValueError("consumed character position exceeds the supplied stream")


def windows(tokenized: list[tuple[Article, list[int]]], *, count: int,
            length: int = 512, max_per_article: int = 4,
            exclude_through_row: int = -1, excluded_hashes: set[str] | None = None):
    """One window per eligible article per round, preserving source order."""
    for value in (count, length, max_per_article):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("positive integer window limits required")
    excluded_hashes = excluded_hashes or set()
    eligible = [(doc, ids) for doc, ids in tokenized if doc.start_row > exclude_through_row
                and doc.text_sha256 not in excluded_hashes and len(ids) >= length]
    result = []
    for round_index in range(max_per_article):
        for doc, ids in eligible:
            start = round_index * length
            stop = start + length
            if stop <= len(ids):
                result.append((doc, start, stop, ids[start:stop]))
                if len(result) == count:
                    return result
    raise ValueError(f"only {len(result)} full windows available for fixed count {count}")
