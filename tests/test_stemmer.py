"""Porter-1980 stemmer parity tests against PyStemmer.

PyStemmer's ``Stemmer.Stemmer('english').stemWord`` is the oracle.
Our implementation is the canonical Porter 1980 algorithm (5 steps);
PyStemmer ships Snowball's English stemmer which is Porter2/snowball-english,
not Porter-1980. The two **mostly** agree on lowercase ASCII tokens —
parity is asserted at >= 99% on a 1000-word sample, with deviations
diagnosed in the assertion message.

Case handling: PyStemmer's snowball treats non-lowercase strings as
opaque-ish (only some chars get rewritten). Our stemmer mirrors this:
all-uppercase passthrough, mixed-case passthrough.
"""

from __future__ import annotations

import pytest

import mojo_bm25s
from mojo_bm25s.stem import stem, stem_corpus


# --------------------------------------------------------------------------
# Oracle setup
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def oracle():
    import Stemmer
    return Stemmer.Stemmer("english").stemWord


# --------------------------------------------------------------------------
# Step 1a: -sses/-ies/-ss/-s
# --------------------------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("caresses", "caress"),
    ("ponies", "poni"),
    ("ties", "tie"),
    ("caress", "caress"),
    ("cats", "cat"),
])
def test_step_1a(word, expected):
    assert stem(word) == expected


# --------------------------------------------------------------------------
# Step 1b: -eed/-ed/-ing + post-processing (double-letter, CVC, -at/-bl/-iz)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    # -eed: only if measure > 0; PyStemmer then runs step 5a which
    # strips the final e from agreed→agree→agre.
    ("feed", "feed"),          # 'f' has m=0 → no strip
    ("agreed", "agre"),        # post step 5a removes the e
    # -ed: only if stem has vowel
    ("plastered", "plaster"),
    ("bled", "bled"),           # 'bl' no vowel → unchanged
    # -ing
    ("motoring", "motor"),
    ("sing", "sing"),           # no vowel before -ing in 's'
    # post-processing — note step 5a then strips trailing e from
    # m>1 stems like 'conflate'→'conflat'.
    ("conflated", "conflat"),
    ("troubled", "troubl"),
    ("sized", "size"),          # m=1, 5a keeps the e (no CVC strip applies)
    ("hopping", "hop"),         # CVC double letter (not l/s/z) → drop one
    ("tanned", "tan"),
    ("falling", "fall"),        # ll → keep both (one of l/s/z)
    ("hissing", "hiss"),
    ("fizzed", "fizz"),
    ("failing", "fail"),
    ("filing", "file"),         # m=1, CVC → add e (then 5a keeps it)
])
def test_step_1b(word, expected):
    assert stem(word) == expected


# --------------------------------------------------------------------------
# Step 1c: y → i (when there's a vowel in the stem)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("happy", "happi"),
    ("sky", "sky"),  # 's' has no vowel → leave 'y' alone
])
def test_step_1c(word, expected):
    assert stem(word) == expected


# --------------------------------------------------------------------------
# Step 2: long-suffix rewrites (-ational → -ate, etc.)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    # -ational → -ate → step 5a drops e → -at
    ("relational", "relat"),
    ("conditional", "condit"),
    ("rational", "ration"),
    ("valenci", "valenc"),
    ("hesitanci", "hesit"),
    ("digitizer", "digit"),
    ("conformabli", "conform"),
    ("radicalli", "radic"),
    ("differentli", "differ"),
    ("vileli", "vile"),
    ("analogousli", "analog"),
    ("vietnamization", "vietnam"),
    ("predication", "predic"),
    ("operator", "oper"),
    ("feudalism", "feudal"),
    ("decisiveness", "decis"),
    ("hopefulness", "hope"),
    ("callousness", "callous"),
    ("formaliti", "formal"),
    ("sensitiviti", "sensit"),
    ("sensibiliti", "sensibl"),
])
def test_step_2(word, expected):
    assert stem(word) == expected


# --------------------------------------------------------------------------
# Step 3: -icate, -ative, -alize, -iciti, -ical, -ful, -ness
# --------------------------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("triplicate", "triplic"),
    # 'formative' → step 3 -ative drops to 'form' (PyStemmer behavior).
    ("formative", "form"),
    ("formalize", "formal"),
    ("electriciti", "electr"),
    ("electrical", "electr"),
    ("hopeful", "hope"),
    ("goodness", "good"),
])
def test_step_3(word, expected):
    assert stem(word) == expected


# --------------------------------------------------------------------------
# Step 4: long suffix removal (m > 1)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("revival", "reviv"),
    ("allowance", "allow"),
    ("inference", "infer"),
    ("airliner", "airlin"),
    ("gyroscopic", "gyroscop"),
    ("adjustable", "adjust"),
    ("defensible", "defens"),
    ("irritant", "irrit"),
    ("replacement", "replac"),
    ("adjustment", "adjust"),
    ("dependent", "depend"),
    ("adoption", "adopt"),
    ("homologous", "homolog"),
    # PyStemmer strips -ism even when m=1: 'communism' → 'commun'.
    ("communism", "commun"),
    ("activate", "activ"),
    ("angulariti", "angular"),
    ("effective", "effect"),
    ("bowdlerize", "bowdler"),
])
def test_step_4(word, expected):
    assert stem(word) == expected


# --------------------------------------------------------------------------
# Step 5a (-e), 5b (-ll → -l)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("probate", "probat"),       # m > 1 → drop e
    ("rate", "rate"),            # m = 1, CVC where second is e... keeps e
    ("cease", "ceas"),           # 5a
    ("controll", "control"),     # 5b: m>1 + ll → l
    ("roll", "roll"),            # m=1, no strip
])
def test_step_5(word, expected):
    assert stem(word) == expected


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("", ""),
    ("a", "a"),
    ("is", "is"),
    ("be", "be"),
    ("y", "y"),
    ("AAA", "AAA"),       # all-caps passthrough (snowball compat)
    ("123", "123"),       # numbers passthrough
])
def test_edge_cases(word, expected):
    assert stem(word) == expected


def test_stem_is_idempotent_on_stems():
    """Once stemmed, restemming should not change it (for common words)."""
    for w in ["cat", "run", "agre", "happi", "format", "electr"]:
        assert stem(stem(w)) == stem(w)


# --------------------------------------------------------------------------
# Corpus helper
# --------------------------------------------------------------------------

def test_stem_corpus_simple():
    docs = [["running", "runs", "ran"]]
    out = stem_corpus(docs)
    assert isinstance(out, list)
    assert len(out) == 1
    assert isinstance(out[0], list)
    # running → run, runs → run, ran → ran (irregular, Porter doesn't handle)
    assert out[0] == ["run", "run", "ran"]


def test_stem_corpus_multiple_docs():
    docs = [
        ["organization", "organized"],
        ["happily", "happy"],
        [],
    ]
    out = stem_corpus(docs)
    assert len(out) == 3
    assert out[2] == []
    # Each doc's tokens get stemmed; per-token result matches `stem(...)`
    for doc_in, doc_out in zip(docs, out):
        assert doc_out == [stem(t) for t in doc_in]


def test_stem_corpus_parity_with_oracle(oracle):
    """Per-token results of `stem_corpus` should equal per-token oracle calls."""
    docs = [
        ["caresses", "ponies", "ties", "caress", "cats"],
        ["feed", "agreed", "plastered", "bled", "motoring"],
        ["conflated", "troubled", "sized", "hopping"],
    ]
    out = stem_corpus(docs)
    for doc_in, doc_out in zip(docs, out):
        expected = [oracle(t) for t in doc_in]
        assert doc_out == expected


# --------------------------------------------------------------------------
# Re-exports
# --------------------------------------------------------------------------

def test_top_level_exports():
    assert mojo_bm25s.stem("running") == "run"
    out = mojo_bm25s.stem_corpus([["cats", "dogs"]])
    assert out == [["cat", "dog"]]


# --------------------------------------------------------------------------
# Parity sample: >= 1000 unique words from a synthetic corpus
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def parity_sample():
    """Build a sample of >=1000 unique lowercase tokens.

    Source: bm25s.tokenize over a synthetic mini-corpus + a curated
    word list. We deliberately use a tokenizer that doesn't itself
    stem (stemmer=None) so we feed raw word forms to the stemmer.
    """
    import bm25s

    # Curated English word list with morphological variants — biased
    # toward forms that exercise Porter's rules (suffixes, double
    # letters, -ies/-ied, etc.).
    seeds = [
        # -s/-es/-ies
        "cats", "dogs", "boxes", "buses", "ponies", "ties", "caresses",
        "kisses", "wishes", "matches", "watches", "babies", "cities",
        "stories", "memories", "histories", "factories", "countries",
        # -ed forms (regular, double-letter, e-final, etc.)
        "walked", "talked", "played", "carried", "married", "studied",
        "hopped", "stopped", "dropped", "tagged", "begged", "planned",
        "hoped", "loved", "lived", "moved", "saved", "tried", "cried",
        # -ing forms
        "running", "swimming", "sitting", "getting", "stopping",
        "hopping", "tagging", "loving", "hoping", "moving", "writing",
        "making", "taking", "having", "doing", "going", "being",
        # -ly
        "quickly", "slowly", "carefully", "happily", "easily", "really",
        "fully", "finally", "exactly", "actually", "usually", "clearly",
        "totally", "personally", "naturally", "directly",
        # -ation / -ization
        "creation", "formation", "relation", "nation", "station",
        "organization", "civilization", "realization", "specialization",
        "characterization", "modernization", "centralization",
        # -ness
        "happiness", "sadness", "kindness", "softness", "darkness",
        "fitness", "weakness", "weariness", "calmness",
        # -ment
        "movement", "agreement", "argument", "department", "treatment",
        "improvement", "development", "judgement",
        # -able / -ible
        "movable", "readable", "writable", "lovable", "drinkable",
        "visible", "invisible", "responsible", "possible", "edible",
        # -ous / -ious
        "famous", "dangerous", "various", "previous", "obvious",
        "curious", "serious", "religious", "delicious", "anxious",
        # -ity / -iti (snowball normalizes ity→iti early)
        "ability", "activity", "quality", "reality", "society",
        "university", "community", "responsibility", "possibility",
        # -ful
        "beautiful", "wonderful", "helpful", "useful", "careful",
        "powerful", "successful", "meaningful", "thankful",
        # -er / -or
        "teacher", "writer", "player", "runner", "swimmer", "singer",
        "actor", "doctor", "director", "creator", "instructor",
        # Various roots
        "agree", "argue", "answer", "appear", "attack", "begin",
        "believe", "build", "buy", "change", "consider", "decide",
        "describe", "develop", "discover", "draw", "explain", "feel",
        "follow", "happen", "include", "increase", "introduce", "learn",
        "leave", "listen", "live", "look", "lose", "love", "mean",
        "meet", "need", "offer", "open", "pay", "play", "produce",
        "provide", "raise", "reach", "read", "receive", "remember",
        "report", "return", "say", "see", "seem", "sell", "send",
        "serve", "set", "show", "speak", "spend", "stand", "start",
        "stay", "stop", "study", "suggest", "take", "talk", "teach",
        "tell", "think", "try", "turn", "understand", "use", "wait",
        "walk", "want", "watch", "win", "work", "write",
        # Common nouns (singular + plural)
        "house", "houses", "school", "schools", "place", "places",
        "system", "systems", "program", "programs", "question",
        "questions", "person", "people", "child", "children", "thing",
        "things", "world", "country", "countries", "family", "families",
        "company", "companies", "service", "services", "business",
        "businesses", "process", "processes", "issue", "issues",
        "policy", "policies", "subject", "subjects", "report", "reports",
        # Adjective comparatives/superlatives
        "happier", "happiest", "easier", "easiest", "faster", "fastest",
        "older", "oldest", "younger", "youngest", "bigger", "biggest",
        "smaller", "smallest", "stronger", "strongest", "weaker",
        "weakest", "richer", "richest", "poorer", "poorest", "longer",
        "longest", "shorter", "shortest", "darker", "darkest",
        # Tech / science vocab
        "computer", "computers", "computing", "computed", "machine",
        "machines", "engine", "engines", "engineering", "engineered",
        "algorithm", "algorithms", "algorithmic", "data", "database",
        "databases", "function", "functions", "functional", "method",
        "methods", "model", "models", "modeling", "modeled",
        "process", "processes", "processing", "processed", "processor",
        "network", "networks", "networking", "networked",
        "analyze", "analyzed", "analyzing", "analysis", "analyses",
        "synthesize", "synthesized", "synthesizing", "synthesis",
        "optimize", "optimized", "optimizing", "optimization",
        "normalize", "normalized", "normalizing", "normalization",
        "minimize", "minimized", "minimizing", "minimization",
        "maximize", "maximized", "maximizing", "maximization",
        # Generic action verbs
        "running", "jumping", "swimming", "diving", "throwing",
        "catching", "hitting", "kicking", "passing", "shooting",
        "scoring", "winning", "losing", "playing", "training",
        "coaching", "teaching", "learning", "studying", "reading",
        "writing", "drawing", "painting", "singing", "dancing",
        # More -ed
        "completed", "started", "ended", "opened", "closed", "moved",
        "removed", "added", "deleted", "updated", "created", "destroyed",
        "selected", "chosen", "picked", "rejected", "accepted", "denied",
        # More -ing
        "creating", "destroying", "selecting", "rejecting", "accepting",
        "denying", "removing", "adding", "deleting", "updating",
        "completing", "starting", "ending", "opening", "closing",
        # ies / ied / ying
        "fly", "flying", "flies", "flew", "flown",
        "cry", "crying", "cries", "cried",
        "try", "trying", "tries", "tried",
        "dry", "drying", "dries", "dried",
        "spy", "spying", "spies", "spied",
        # Various -er roots
        "manager", "managers", "management", "managing", "managed",
        "designer", "designers", "designing", "designed", "design",
        "developer", "developers", "developing", "developed",
        "publisher", "publishers", "publishing", "published",
        # Politics / society
        "government", "governments", "governance", "governing",
        "election", "elections", "electing", "elected", "elector",
        "voting", "voters", "voted", "vote", "votes",
        "democracy", "democratic", "democratically",
        "republic", "republican", "republicanism",
        "freedom", "freedoms", "free", "freely", "freedom",
        "liberty", "liberties", "liberal", "liberalism", "liberally",
        # Education
        "education", "educational", "educate", "educated", "educator",
        "school", "schooling", "schooled", "schools",
        "student", "students", "study", "studying", "studied", "studies",
        "research", "researcher", "researchers", "researching",
        "researched",
        # Common adverbs / connectives (mostly passthrough)
        "the", "and", "but", "or", "for", "with", "from", "into",
        "about", "above", "below", "before", "after", "during", "while",
        "since", "until", "through", "across", "around", "between",
    ]
    # Use bm25s tokenize as the realistic preprocessing step
    tokens = bm25s.tokenize(
        " ".join(seeds), stopwords=None, stemmer=None, return_ids=False,
    )
    # bm25s.tokenize returns a list of lists (one per "doc")
    flat = []
    for doc in tokens:
        flat.extend(doc)
    flat.extend(seeds)  # keep originals too in case tokenize merged anything

    # Augment with scifact corpus tokens to reach >= 1000 unique words —
    # the curated seed list alone gives ~500 unique forms; scifact's
    # technical English brings the rest. Loader is cached after first
    # call so this stays cheap in CI.
    import sys
    from pathlib import Path
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    try:
        from benchmarks.datasets import load_beir
        ds = load_beir("scifact", queries_subsample=10)
        for doc in ds.corpus_tokens():
            flat.extend(doc)
    except Exception:
        # If scifact unavailable (no network, no cache), fall back to
        # generating variants from the seed list.
        pass

    unique = list(dict.fromkeys(flat))
    # filter to lowercase ASCII tokens of length >= 1
    unique = [t for t in unique if t.isascii() and t.islower() and t]
    assert len(unique) >= 1000, (
        f"need >=1000 unique tokens for parity, got {len(unique)}; "
        f"extend the seed list or ensure scifact is available"
    )
    return unique[:1500]  # cap for runtime


def test_parity_sample_agreement(parity_sample, oracle):
    """Per-token agreement with PyStemmer on the parity sample must be >=99%."""
    mismatches: list[tuple[str, str, str]] = []
    for tok in parity_sample:
        got = stem(tok)
        exp = oracle(tok)
        if got != exp:
            mismatches.append((tok, got, exp))

    n = len(parity_sample)
    agreement = 1.0 - (len(mismatches) / n)
    # Show first 20 mismatches in the message
    sample_msg = "\n  ".join(
        f"{tok!r}: ours={got!r}, oracle={exp!r}"
        for tok, got, exp in mismatches[:20]
    )
    assert agreement >= 0.99, (
        f"PyStemmer parity {agreement:.3%} (< 99%); "
        f"{len(mismatches)}/{n} tokens diverged.\n  "
        f"First mismatches:\n  {sample_msg}"
    )
