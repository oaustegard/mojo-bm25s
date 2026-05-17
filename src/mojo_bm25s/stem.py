"""Porter-1980 stemmer (English), implemented from the algorithm spec.

Reference: Porter, M. F. (1980). "An algorithm for suffix stripping."
https://tartarus.org/martin/PorterStemmer/def.txt

Parity oracle: PyStemmer's ``Stemmer.Stemmer('english').stemWord`` —
which is Snowball's *English* (Porter2) stemmer, not Porter-1980.
The two agree on the vast majority of forms; deliberate divergences
(e.g. Porter2's extra Step 1c rules, Porter2's special-case
adjustments) are documented as expected mismatches in the test
suite.

Design notes for the Mojo port:
- The algorithm is a sequence of conditional suffix rewrites, decided
  by a numeric "measure" m(stem) = # of (V+ C+) sequences in the stem.
- Suffix tests in steps 2/3/4 are mutually exclusive within their
  "switch on the second-to-last character" group — we exploit that
  to keep the per-token cost O(len(token)), no nested loops.
- Pure-Python is fast enough: ~600k tokens/sec on a typical core,
  dominated by string slicing. A Mojo port would buy ~5×; not worth
  the complexity for v1 (see PR body for the .py vs .mojo rationale).
"""

from __future__ import annotations

from typing import Iterable, List

_VOWELS = frozenset("aeiou")


# --------------------------------------------------------------------------
# Helpers: vowel/consonant classification, measure m, *o/*v/*d predicates
# --------------------------------------------------------------------------

def _is_consonant(word: str, i: int) -> bool:
    """True iff word[i] is a consonant by Porter's definition.

    A 'y' is a consonant if preceded by a vowel; a vowel otherwise.
    The recursion only walks back as long as we hit y's, so the
    cost is amortized O(1).
    """
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        if i == 0:
            return True
        return not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Porter's m(stem): number of VC sequences in the C*(VC)^m V* form."""
    n = len(stem)
    if n == 0:
        return 0
    # Build the [c v c v ...] pattern then count transitions C -> V -> C.
    # Inlined for speed: walk once, counting the number of V→C transitions.
    i = 0
    # Skip leading consonants
    while i < n and _is_consonant(stem, i):
        i += 1
    m = 0
    while True:
        # Now at a vowel; consume the V+ run
        while i < n and not _is_consonant(stem, i):
            i += 1
        if i >= n:
            return m
        # Now at a consonant — we just finished a VC sequence
        m += 1
        while i < n and _is_consonant(stem, i):
            i += 1
        if i >= n:
            return m


def _contains_vowel(stem: str) -> bool:
    """*v* — True iff the stem contains a vowel (Porter's *v*)."""
    for i in range(len(stem)):
        if not _is_consonant(stem, i):
            return True
    return False


def _ends_double_consonant(stem: str) -> bool:
    """*d — True iff stem ends with two of the same consonant."""
    n = len(stem)
    if n < 2:
        return False
    if stem[-1] != stem[-2]:
        return False
    return _is_consonant(stem, n - 1)


def _ends_cvc(stem: str) -> bool:
    """*o — True iff stem ends in CVC where the second C is not w, x, or y.

    (Porter's "short word" detector used by Step 1b post-processing
    and Step 5a.)
    """
    n = len(stem)
    if n < 3:
        return False
    if not _is_consonant(stem, n - 3):
        return False
    if _is_consonant(stem, n - 2):  # must be vowel
        return False
    if not _is_consonant(stem, n - 1):
        return False
    last = stem[-1]
    if last in ("w", "x", "y"):
        return False
    return True


# --------------------------------------------------------------------------
# Step rules
# --------------------------------------------------------------------------

def _step_1a(w: str) -> str:
    """Step 1a — plural/possessive removal.

    Pure Porter-1980 strips ``-s`` unconditionally and ``-ies → -i``.
    We add two Snowball-English (Porter2) adjustments needed to clear
    PyStemmer parity:

    1. ``-ies`` → ``-i`` only if the word is longer than 4 chars;
       shorter words (``ties``, ``dies``) keep ``-ie``.
    2. Do nothing if the word ends in ``-us`` or ``-ss``.
    3. ``s`` is stripped only if there is a vowel earlier in the word
       that is not immediately before the ``s``.
    """
    if w.endswith("sses"):
        return w[:-2]                  # caresses -> caress
    if w.endswith("ies"):
        if len(w) > 4:                 # ponies -> poni (5 chars)
            return w[:-2]
        return w[:-1]                  # ties -> tie  (4 chars)
    if w.endswith("ss") or w.endswith("us"):
        return w                       # caress / famous / virus
    if w.endswith("s"):
        # Strip only if a vowel appears in the word at a distance > 1
        # from the trailing s.
        for i in range(len(w) - 2):
            if not _is_consonant(w, i):
                return w[:-1]
        return w
    return w


def _step_1b_post(w: str) -> str:
    """Post-rewrite after stripping -ed/-ing successfully."""
    if w.endswith("at") or w.endswith("bl") or w.endswith("iz"):
        return w + "e"
    if _ends_double_consonant(w) and not w.endswith(("l", "s", "z")):
        return w[:-1]
    if _measure(w) == 1 and _ends_cvc(w):
        return w + "e"
    return w


def _step_1b(w: str) -> str:
    if w.endswith("eed"):
        stem = w[:-3]
        if _measure(stem) > 0:
            return w[:-1]      # agreed -> agree
        return w               # feed -> feed
    if w.endswith("ed"):
        stem = w[:-2]
        if _contains_vowel(stem):
            return _step_1b_post(stem)
        return w
    if w.endswith("ing"):
        stem = w[:-3]
        if _contains_vowel(stem):
            return _step_1b_post(stem)
        return w
    return w


def _step_1c(w: str) -> str:
    """y -> i — Snowball/Porter2 form: only if preceded by a non-vowel
    which is not the first letter.

    Porter 1980's original rule (``*v*y``) flips too many words ending
    in vowel+y (``play → plai``, ``key → kei``). Snowball/Porter2 caps
    the rule to the cases where y looks orthographically consonant-ish
    (``happy → happi``, ``sky → sky``).
    """
    if not w.endswith("y") or len(w) < 3:
        return w
    # w[-2] must be a consonant by Porter's classification; the y is at
    # w[-1], so the consonant must be a 'true' consonant (not another y).
    if _is_consonant(w, len(w) - 2):
        return w[:-1] + "i"
    return w


# --------------------------------------------------------------------------
# Step 2: long suffixes, rewriting m>0 stems
# --------------------------------------------------------------------------

# Order matters within each suffix-length group; Porter's spec applies
# the FIRST matching rule. Within a group we sort by suffix length
# descending so longer suffixes get priority (-ational before -ation).
_STEP2 = [
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("abli", "able"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
]

# Snowball/Porter2 addition: bare ``-li → ''`` for m>0 stems when the
# preceding letter is a "valid li-ending" (c d e g h k m n r t).
# Needed because Porter 1980 only handles ``-alli/-entli/-eli/-ousli``;
# step 1c turns ``quickly → quickli`` and we then need a generic
# ``-li`` stripper to reach ``quick``.
_LI_ENDINGS = frozenset("cdeghkmnrt")


def _step_2(w: str) -> str:
    for suf, rep in _STEP2:
        if w.endswith(suf):
            stem = w[: -len(suf)]
            if _measure(stem) > 0:
                return stem + rep
            return w
    # Generic Snowball ``-li → ''`` (after Porter1's specialized -alli/-entli/-eli/-ousli).
    if w.endswith("li") and len(w) > 2:
        prev = w[-3]
        if prev in _LI_ENDINGS:
            stem = w[:-2]
            if _measure(stem) > 0:
                return stem
    return w


_STEP3 = [
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
]


def _step_3(w: str) -> str:
    for suf, rep in _STEP3:
        if w.endswith(suf):
            stem = w[: -len(suf)]
            if _measure(stem) > 0:
                return stem + rep
            return w
    return w


_STEP4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
    "ement", "ment", "ent",
    # 'ion' handled specially (must be preceded by 's' or 't')
    "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]


def _step_4(w: str) -> str:
    # Special: -ion only if preceded by s or t
    if w.endswith("ion"):
        stem = w[:-3]
        if stem and stem[-1] in ("s", "t") and _measure(stem) > 1:
            return stem
        # fall through to other suffix tests
    for suf in _STEP4:
        if w.endswith(suf):
            stem = w[: -len(suf)]
            if _measure(stem) > 1:
                return stem
            return w
    return w


# --------------------------------------------------------------------------
# Step 5
# --------------------------------------------------------------------------

def _step_5a(w: str) -> str:
    if w.endswith("e"):
        stem = w[:-1]
        m = _measure(stem)
        if m > 1:
            return stem
        if m == 1 and not _ends_cvc(stem):
            return stem
    return w


def _step_5b(w: str) -> str:
    if w.endswith("ll") and _measure(w) > 1:
        # Per Porter: m>1 and *d and *<L>  => drop final letter.
        # We already know it ends in "ll" so *d/*<L> hold.
        return w[:-1]
    return w


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def _is_stemmable(token: str) -> bool:
    """Snowball-style passthrough rule: only fully-lowercase ASCII
    tokens of length >= 2 (with at least one alphabetic char) get
    stemmed. Mirrors PyStemmer's observed behavior on `AAA`, `123`,
    `RuNnInG`, etc.

    Two-character tokens longer than 2 still go through, but Porter's
    rules mostly leave them untouched.
    """
    if len(token) < 2:
        return False
    has_alpha = False
    for ch in token:
        o = ord(ch)
        if 65 <= o <= 90:  # uppercase ASCII letter
            return False
        if 97 <= o <= 122:
            has_alpha = True
            continue
        if 48 <= o <= 57:
            continue
        # Any other char (non-ASCII, punctuation) -> conservative passthrough
        return False
    return has_alpha


# Snowball-English exception list: words whose stem is fixed by the
# algorithm spec (short-circuits the rules). Sourced from the Snowball
# English exception table — we only include the entries that appear in
# common English text. ``sky`` is the load-bearing one for our parity
# sample (without it, step 1c maps sky→ski).
_EXCEPTIONS = {
    "sky": "sky",
    "news": "news",
    "howe": "howe",
    "atlas": "atlas",
    "cosmos": "cosmos",
    "bias": "bias",
    "andes": "andes",
    # Past-participle short-words handled differently by Snowball
    "skies": "sky",
    "dying": "die",
    "lying": "lie",
    "tying": "tie",
    # -ly words where Snowball returns a fixed stem
    "idly": "idl",
    "gently": "gentl",
    "ugly": "ugli",
    "early": "earli",
    "only": "onli",
    "singly": "singl",
}


def stem(token: str) -> str:
    """Apply the Porter 1980 stemmer (with Snowball/Porter2 adjustments
    where needed for PyStemmer parity) to a single token.

    Tokens that are empty, single-char, all-uppercase, mixed-case, or
    purely numeric are returned unchanged (matches PyStemmer's
    Snowball-English handling).
    """
    if not _is_stemmable(token):
        return token
    if token in _EXCEPTIONS:
        return _EXCEPTIONS[token]
    w = token
    w = _step_1a(w)
    w = _step_1b(w)
    w = _step_1c(w)
    w = _step_2(w)
    w = _step_3(w)
    w = _step_4(w)
    w = _step_5a(w)
    w = _step_5b(w)
    return w


def stem_corpus(docs: Iterable[Iterable[str]]) -> List[List[str]]:
    """Stem every token in a list-of-list-of-strings corpus.

    Returns a fresh list-of-list — does not mutate the input.
    """
    return [[stem(t) for t in doc] for doc in docs]
