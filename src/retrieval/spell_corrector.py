"""
Spell correction and autocomplete module for e-commerce search queries.

This module provides spelling correction using Levenshtein edit distance,
corpus-derived word frequencies, TextBlob fallback, and prefix-based query auto-completion.
"""

from collections import Counter
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import nltk
from nltk.tokenize import word_tokenize

try:
    from textblob import Word as TextBlobWord
except ImportError:
    TextBlobWord = None


class SpellCorrector:
    """
    Spelling corrector and completion suggester for search queries.
    
    Combines vocabulary term frequencies, Levenshtein edit distance,
    and TextBlob fallback for robust spell correction and prefix matching.
    """

    def __init__(
        self,
        vocabulary: Optional[Union[Dict[str, int], Iterable[str], Counter]] = None,
    ) -> None:
        """
        Initialize the SpellCorrector with an optional vocabulary.

        Args:
            vocabulary: A dictionary/Counter of word frequencies or an iterable of words.
        """
        self.vocab_counts: Counter = Counter()

        if vocabulary is not None:
            if isinstance(vocabulary, dict):
                self.vocab_counts.update(vocabulary)
            elif isinstance(vocabulary, (list, tuple, set)):
                # If elements contain spaces, tokenize or split; otherwise count words
                for item in vocabulary:
                    if isinstance(item, str):
                        words = re.findall(r"\b[a-zA-Z0-9'-]+\b", item.lower())
                        if words:
                            self.vocab_counts.update(words)
                        else:
                            self.vocab_counts[item.lower()] += 1

    def build_vocabulary(self, documents: Sequence[str]) -> None:
        """
        Extract unique words and their frequencies from a corpus of documents.

        Args:
            documents: List or sequence of text documents.
        """
        for doc in documents:
            if not isinstance(doc, str) or not doc.strip():
                continue
            try:
                tokens = word_tokenize(doc.lower())
            except LookupError:
                tokens = re.findall(r"\b[a-zA-Z0-9'-]+\b", doc.lower())

            words = [t for t in tokens if re.match(r"^[a-zA-Z0-9'-]+$", t)]
            self.vocab_counts.update(words)

    def _edit_distance(self, word1: str, word2: str) -> int:
        """
        Compute Levenshtein edit distance between two strings.

        Args:
            word1: First string.
            word2: Second string.

        Returns:
            The minimum number of single-character edits (insertions,
            deletions, substitutions) required to transform word1 into word2.
        """
        m, n = len(word1), len(word2)
        if m == 0:
            return n
        if n == 0:
            return m

        # Use two rows to save space
        previous_row = list(range(n + 1))
        current_row = [0] * (n + 1)

        for i in range(1, m + 1):
            current_row[0] = i
            c1 = word1[i - 1]
            for j in range(1, n + 1):
                c2 = word2[j - 1]
                cost = 0 if c1 == c2 else 1
                current_row[j] = min(
                    previous_row[j] + 1,        # deletion
                    current_row[j - 1] + 1,       # insertion
                    previous_row[j - 1] + cost    # substitution
                )
            previous_row = list(current_row)

        return current_row[n]

    def _candidates(self, word: str, max_distance: int = 2) -> List[str]:
        """
        Generate candidate corrections within the specified edit distance.

        Args:
            word: The query word to correct.
            max_distance: Maximum allowable Levenshtein distance.

        Returns:
            List of candidate words from vocabulary ordered by edit distance
            and frequency.
        """
        cleaned = word.lower().strip()
        if not cleaned:
            return []

        # If word is already valid in vocabulary
        if cleaned in self.vocab_counts and self.vocab_counts[cleaned] > 0:
            return [cleaned]

        candidates: List[Tuple[int, int, str]] = []
        len_word = len(cleaned)

        for candidate, count in self.vocab_counts.items():
            if abs(len(candidate) - len_word) > max_distance:
                continue
            dist = self._edit_distance(cleaned, candidate)
            if dist <= max_distance:
                # Store (-count) for descending order sort
                candidates.append((dist, -count, candidate))

        candidates.sort()
        return [cand for _, _, cand in candidates]

    def correct_word(self, word: str) -> str:
        """
        Return the most likely correction for a single word based on frequency.

        Falls back to TextBlob spelling correction if no vocabulary candidate is found.

        Args:
            word: Word to be corrected.

        Returns:
            Corrected word string, preserving casing where possible.
        """
        if not word or not word.strip():
            return word

        # Preserve punctuation tokens as-is
        if not re.search(r"[a-zA-Z0-9]", word):
            return word

        cleaned = word.lower().strip()

        # Exact match in vocabulary
        if cleaned in self.vocab_counts and self.vocab_counts[cleaned] > 0:
            return self._match_case(word, cleaned)

        # Check candidate corrections
        candidates = self._candidates(cleaned, max_distance=2)
        if candidates:
            return self._match_case(word, candidates[0])

        # TextBlob fallback
        if TextBlobWord is not None:
            try:
                tb_word = TextBlobWord(cleaned)
                corrected_tb = str(tb_word.correct())
                if corrected_tb and corrected_tb != cleaned:
                    return self._match_case(word, corrected_tb)
            except Exception:
                pass

        return word

    def correct_query(self, query: str) -> str:
        """
        Correct each word in the query using NLTK word tokenization.

        Args:
            query: User search query string.

        Returns:
            Spelling-corrected query string.
        """
        if not query or not query.strip():
            return query

        try:
            tokens = word_tokenize(query)
        except LookupError:
            tokens = query.split()

        corrected_tokens = []
        for token in tokens:
            if re.search(r"[a-zA-Z0-9]", token):
                corrected_tokens.append(self.correct_word(token))
            else:
                corrected_tokens.append(token)

        # Reconstruct query while handling punctuation cleanly
        corrected_query = ""
        for i, token in enumerate(corrected_tokens):
            if i == 0 or token in {".", ",", "!", "?", ";", ":", "'", '"', ")", "]", "}"}:
                corrected_query += token
            elif corrected_tokens[i - 1] in {"(", "[", "{"}:
                corrected_query += token
            else:
                corrected_query += " " + token

        return corrected_query.strip()

    def suggest_completions(self, prefix: str, top_k: int = 5) -> List[str]:
        """
        Generate prefix-based autocomplete suggestions from vocabulary.

        Args:
            prefix: Starting prefix string.
            top_k: Number of suggestions to return.

        Returns:
            Top-k vocabulary terms matching the prefix, sorted by corpus frequency.
        """
        prefix_clean = prefix.lower().strip()
        if not prefix_clean:
            return []

        matches = [
            (candidate, count)
            for candidate, count in self.vocab_counts.items()
            if candidate.startswith(prefix_clean)
        ]

        # Sort descending by frequency, then ascending alphabetically
        matches.sort(key=lambda item: (-item[1], item[0]))
        return [word for word, _ in matches[:top_k]]

    @staticmethod
    def _match_case(original: str, corrected: str) -> str:
        """Helper to match the case of the corrected word to the original word."""
        if original.isupper() and len(original) > 1:
            return corrected.upper()
        if original.istitle():
            return corrected.capitalize()
        return corrected
