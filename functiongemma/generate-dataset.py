#!/usr/bin/env python3
"""
Generate the FunctionGemma fine-tuning dataset for Ari.

Combines:
1. Ari built-in skill paraphrases (extracted from ari-engine via the
   `export-utterances` Cargo binary — single source of truth)
2. Community skill examples (parsed from ari-skills SKILL.md manifests)
3. Negative examples (general knowledge, should NOT trigger any function),
   scaled to the positive count so abstention stays well represented.

Every training utterance is normalised through the engine's own
`normalize_input` (via the `normalize` bin) before sample generation: the
router is only ever served normalised text at inference, so training on raw
text trains it on a distribution it never sees. Tool-call args and skill
descriptions are schema, not utterances, and are deliberately NOT normalised.

We deliberately do NOT use Google's mobile-actions demo dataset: the base
model (`google/functiongemma-270m-it`) is already function-calling pretrained,
those are Android actions Ari doesn't support, and its 9,654 always-call
samples drowned the negatives — the cause of the r70→r75 abstention regression.

Output: JSONL on stdout in the same format as google/mobile-actions,
ready for SFTTrainer fine-tuning with FunctionGemma's chat template.

Usage:
    python3 generate-dataset.py [--locale <xx>] > dataset.jsonl

    --locale defaults to "en".

Environment variables:
    ARI_ENGINE_DIR  — path to ari-engine checkout (auto-discovered if not set)
    ARI_SKILLS_DIR  — path to ari-skills checkout (auto-discovered if not set)
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

random.seed(42)

# ── Locate ari-engine and run the export binary ────────────────────────

def find_engine_dir() -> Path:
    """Find ari-engine relative to this script or via env var."""
    if env := os.environ.get("ARI_ENGINE_DIR"):
        p = Path(env)
        if (p / "Cargo.toml").is_file():
            return p
        sys.exit(f"ERROR: ARI_ENGINE_DIR={env} doesn't contain Cargo.toml")

    # Look for sibling clones in common locations
    here = Path(__file__).resolve().parent
    candidates = [
        # Sibling under same parent (e.g. ~/projects/ari-tools and ~/projects/ari-engine)
        here.parent.parent / "ari-engine",
        # Sibling under the local Ari workspace
        here.parent.parent.parent / "ari-engine",
        # Same directory level
        here.parent / "ari-engine",
    ]
    for c in candidates:
        if (c / "Cargo.toml").is_file():
            return c

    sys.exit(
        "ERROR: could not find ari-engine. Set ARI_ENGINE_DIR=/path/to/ari-engine "
        "or clone it as a sibling of ari-tools."
    )


def export_skills(engine_dir: Path, locale: str) -> list:
    """Run `export-utterances --locale <locale>` and parse JSON."""
    print(f"Exporting {locale} skills from {engine_dir}...", file=sys.stderr)
    result = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "ari-skills", "--bin",
         "export-utterances", "--", "--locale", locale],
        cwd=engine_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def normalize_texts(engine_dir: Path, texts: list, locale: str) -> list:
    """Normalise training text through the engine's OWN normalize_input.

    The router is served normalize_input()'d text at inference, so the corpus
    must be normalised the same way or we train on a distribution the model
    never sees. We shell out to the `normalize` bin rather than reimplement
    it here: a Python replica would drift, and would be a second source of
    truth for the exact function that defines train/serve parity.

    One subprocess for the whole batch — not one per string.
    """
    if not texts:
        return []
    bad = [t for t in texts if "\n" in t]
    if bad:
        sys.exit(f"ERROR: training text contains a newline, which breaks the "
                 f"line-per-text protocol: {bad[:3]}")
    result = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "ari-skills", "--bin",
         "normalize", "--", "--locale", locale],
        cwd=engine_dir,
        input="\n".join(texts),
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout.split("\n")
    if out and out[-1] == "":
        out.pop()  # trailing newline from println!
    if len(out) != len(texts):
        sys.exit(f"ERROR: normalize returned {len(out)} lines for {len(texts)} "
                 f"inputs — the corpus would be silently mis-paired")
    return out


def keyword_hits(engine_dir: Path, texts: list, locale: str,
                 skills_dir: Path | None = None) -> list:
    """Ask the engine which of these utterances the KEYWORD scorer already wins.

    The router is the fallback tier — it only runs when the keyword scorer
    finds nothing. An example the scorer claims is one the router never sees
    in production, so training on it spends 270M-model capacity on a case it
    is never asked about.

    Same single-source-of-truth rule as normalize_texts: we shell out to the
    engine rather than reimplement the scorer, because a Python replica would
    drift from the ranking logic that actually decides this in production.

    `skills_dir` is the ari-skills checkout whose `skills/` root holds the
    community manifests. Passing it is what makes the oracle honest: without
    it the engine registers only the six built-ins, so a community skill's own
    `matching.patterns` are never in the running and its examples can never be
    detected as keyword-hits — even the ones its own patterns win outright in
    production. Those examples then sit in the corpus as pure waste. Pass None
    only if the checkout is genuinely unavailable.

    Note: pass RAW text. keyword_decision() normalises internally, exactly as
    production does with raw user input.

    One subprocess for the whole batch — not one per string.

    Built with --no-default-features: ari-ffi defaults to the `llm` feature,
    which pulls llama-cpp-sys and needs libclang at build time. The training
    container ships no clang, so the default build would fail here before a
    single row of the dataset was written. The keyword scorer doesn't touch
    the LLM, so dropping the feature costs this oracle nothing. Registering
    the community skills stays inside that light build: the loader (and the
    wasmtime it uses to compile skill modules) is already an unconditional
    dependency of ari-ffi, so --skills-dir adds no new build requirement.
    """
    if not texts:
        return []
    bad = [t for t in texts if "\n" in t]
    if bad:
        sys.exit(f"ERROR: training text contains a newline, which breaks the "
                 f"line-per-text protocol: {bad[:3]}")
    cmd = ["cargo", "run", "--quiet", "-p", "ari-ffi", "--no-default-features",
           "--bin", "keyword-hit", "--", "--locale", locale]
    if skills_dir is not None:
        cmd += ["--skills-dir", str(skills_dir / "skills")]
    result = subprocess.run(
        cmd,
        cwd=engine_dir,
        input="\n".join(texts),
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout.split("\n")
    if out and out[-1] == "":
        out.pop()  # trailing newline from println!
    if len(out) != len(texts):
        sys.exit(f"ERROR: keyword-hit returned {len(out)} verdicts for "
                 f"{len(texts)} inputs — the corpus would be silently mis-paired")
    return [line == "true" for line in out]


# ── Locate ari-skills and parse community SKILL.md files ─────────────

def find_skills_dir() -> Path | None:
    """Find ari-skills relative to this script or via env var."""
    if env := os.environ.get("ARI_SKILLS_DIR"):
        p = Path(env)
        if (p / "skills").is_dir():
            return p
        print(f"WARNING: ARI_SKILLS_DIR={env} has no skills/ dir, skipping community skills", file=sys.stderr)
        return None

    here = Path(__file__).resolve().parent
    candidates = [
        here.parent.parent / "ari-skills",
        here.parent.parent.parent / "ari-skills",
        here.parent / "ari-skills",
    ]
    for c in candidates:
        if (c / "skills").is_dir():
            return c

    print("WARNING: could not find ari-skills, skipping community skills", file=sys.stderr)
    return None


def parse_skillfile_yaml(path: Path) -> dict | None:
    """Minimal YAML frontmatter parser for SKILL.md. Extracts metadata.ari fields."""
    # Deliberately no fallback. Without PyYAML every community skill silently
    # disappears from the corpus AND from check_banks' drift detection — the
    # generator still "succeeds", it just trains on the six built-ins. Crashing
    # is the kind thing to do here.
    import yaml

    text = path.read_text()
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end < 0:
        return None
    frontmatter = text[3:end].strip()
    try:
        doc = yaml.safe_load(frontmatter)
    except Exception:
        return None
    return doc


def load_community_skills(skills_dir: Path, locale: str) -> list:
    """Walk skills/*/SKILL.<locale>.md and extract id, description, params, examples."""
    skills_root = skills_dir / "skills"
    if not skills_root.is_dir():
        return []

    community = []
    for skill_dir in sorted(skills_root.iterdir()):
        # Prefer the requested locale's manifest; fall back to English, then
        # to the pre-migration SKILL.md for any skill not yet migrated.
        manifest = skill_dir / f"SKILL.{locale}.md"
        if not manifest.is_file():
            manifest = skill_dir / "SKILL.en.md"
        if not manifest.is_file():
            manifest = skill_dir / "SKILL.md"
        if not manifest.is_file():
            continue

        doc = parse_skillfile_yaml(manifest)
        if not doc:
            continue

        ari = (doc.get("metadata") or {}).get("ari")
        if not ari:
            continue

        # Skip assistant skills — they don't enter routing
        if ari.get("type") == "assistant":
            continue

        skill_id = ari.get("id")
        description = doc.get("description", "")
        if not skill_id or not description:
            continue

        # Parse examples
        raw_examples = ari.get("examples", [])
        if not raw_examples:
            continue

        examples = []
        for ex in raw_examples:
            text = ex.get("text") if isinstance(ex, dict) else None
            if not text:
                continue
            args = ex.get("args", {}) if isinstance(ex, dict) else {}
            if args is None:
                args = {}
            examples.append({"text": text, "args": args})

        if not examples:
            continue

        # Build a parameters schema from the examples' args keys
        # (simple heuristic — if any example has args, infer string properties)
        params = {"type": "object", "properties": {}}
        for ex in examples:
            for key in ex["args"]:
                if key not in params["properties"]:
                    params["properties"][key] = {"type": "string"}
        if params["properties"]:
            params["required"] = list(params["properties"].keys())

        community.append({
            "id": skill_id,
            "description": description,
            "parameters": params,
            "examples": examples,
        })

    return community


# ── Negative examples (should NOT match any skill) ─────────────────────

NEGATIVE_EXAMPLES = [
    "what is the capital of France",
    "who wrote Romeo and Juliet",
    "why is the sky blue",
    "tell me a joke",
    "how far is the moon from earth",
    "what's the meaning of life",
    "who is the president of the United States",
    "how many continents are there",
    "what's the speed of light",
    "explain quantum physics",
    "what is photosynthesis",
    "who painted the Mona Lisa",
    "what language do they speak in Brazil",
    "how many bones are in the human body",
    "what's the tallest building in the world",
    "when was the internet invented",
    "who discovered gravity",
    "what causes earthquakes",
    "how does wifi work",
    "what is the biggest ocean",
    "tell me about dinosaurs",
    "what's the boiling point of water",
    "who is Albert Einstein",
    "what year did world war 2 end",
    "how do birds fly",
    "what is DNA",
    "who invented the light bulb",
    "what is climate change",
    "how many planets are in the solar system",
    "what is the longest river in the world",
    "explain how a car engine works",
    "what is the most spoken language",
    "who was the first person on the moon",
    "what does AI stand for",
    "how do vaccines work",
    "what is the smallest country in the world",
    "who composed the four seasons",
    "what is blockchain",
    "how do airplanes stay in the air",
    "what is the great wall of china",
    "tell me something interesting",
    "what's a good book to read",
    "recommend a movie",
    "what should I have for dinner",
    "I'm bored",
    "do you like music",
    "what's your favourite colour",
    "are you a robot",
    "what can you do",
    "thank you",
    "thanks ari",
    "never mind",
    "forget it",
    "that's all",
    "goodbye",
    "see you later",
    "talk to you later",
    "I'm done",
    "nothing",
    "okay",
]

# Programmatically expand the negative set. Without enough negatives the
# router is biased to always emit a function and stops abstaining on
# general-knowledge questions — this is the r70→r75 regression (one bad
# nightly routed "what is the capital of the UAE" straight to a skill).
# The per-skill paraphrases number in the hundreds and grow every time a
# skill is added, so the negatives have to keep pace. Templates are rotated
# per entity so the model learns the INTENT (general knowledge → no
# function), not a fixed sentence shape. NOTHING here may contain a skill
# trigger word (date/time/calculate/open/search/look up/google/find/play/
# remind/weather/timer/alarm/coin) or it would become a false negative.

_CAPITAL_COUNTRIES = [
    "the united arab emirates", "saudi arabia", "south korea", "new zealand",
    "Japan", "Brazil", "Egypt", "Canada", "Australia", "Kenya", "Norway",
    "Thailand", "Mexico", "Peru", "Greece", "Portugal", "Vietnam", "Morocco",
    "Argentina", "Sweden", "Poland", "Turkey", "Indonesia", "Nigeria", "Chile",
    "Finland", "Ireland", "Hungary", "Iceland", "the Philippines",
]

_CAPITAL_TEMPLATES = [
    "what is the capital of {}",
    "what's the capital city of {}",
    "which city is the capital of {}",
    "tell me the capital of {}",
]

_LANGUAGE_COUNTRIES = [
    "Brazil", "Switzerland", "Austria", "Belgium", "Morocco", "the Philippines",
    "Nigeria", "Peru", "Iran", "Kazakhstan", "Finland", "Egypt", "India",
    "Mexico", "Vietnam",
]

_PEOPLE = [
    "Marie Curie", "Nelson Mandela", "Isaac Newton", "Leonardo da Vinci",
    "William Shakespeare", "Cleopatra", "Charles Darwin", "Mahatma Gandhi",
    "Vincent van Gogh", "Nikola Tesla", "Winston Churchill", "Frida Kahlo",
    "Galileo", "Beethoven", "Aristotle", "Napoleon", "Stephen Hawking",
    "Rosa Parks", "Confucius", "Pablo Picasso", "Ada Lovelace", "Genghis Khan",
    "Florence Nightingale", "Alan Turing",
]

_PEOPLE_TEMPLATES = [
    "who is {}",
    "who was {}",
    "tell me about {}",
    "what is {} known for",
]

_CONCEPTS = [
    "gravity", "inflation", "a black hole", "the stock market", "evolution",
    "the greenhouse effect", "the water cycle", "a recession", "nuclear fusion",
    "the big bang", "osmosis", "democracy", "capitalism", "virtual reality",
    "quantum entanglement", "the immune system", "machine learning",
    "supply and demand", "compound interest", "natural selection", "inertia",
    "the theory of relativity",
]

_CONCEPT_TEMPLATES = [
    "what is {}",
    "explain {}",
    "can you explain {}",
    "what does {} mean",
]

_HOW_THINGS = [
    "the human heart", "a refrigerator", "a microwave", "the internet", "GPS",
    "solar panels", "a battery", "the brain", "nuclear power", "radio waves",
    "a jet engine", "the stock market", "electricity", "the human eye",
    "a touchscreen", "noise cancelling headphones",
]

# Hand-written negatives across geography, history and demographics — these
# round out the templated sets with less formulaic phrasing.
_MISC_FACTUAL_NEGATIVES = [
    "what is the highest mountain in the world",
    "what is the deepest part of the ocean",
    "what is the hottest place on earth",
    "what is the largest desert",
    "what is the oldest city in the world",
    "which country has the most people",
    "what year did the berlin wall fall",
    "when did the roman empire collapse",
    "when was the printing press invented",
    "what year did humans first land on the moon",
    "when did the cold war end",
    "what caused the first world war",
    "when did the dinosaurs go extinct",
    "what is the population of Tokyo",
    "how many people live in China",
    "how big is Russia",
    "how large is the Pacific ocean",
    "what is the chemical symbol for gold",
    "how many moons does Jupiter have",
    "what is the freezing point of water",
    "why do we dream",
    "how do tides work",
    "what makes the northern lights",
    "why is the ocean salty",
    "how old is the universe",
    "what is the human body made of",
    "why do leaves change colour",
    "what is absolute zero",
    "how many strings does a violin have",
    "what is the currency of Japan",
]


# ── Italian negatives (should NOT match any skill) ─────────────────────
# Same job as the English pool: teach the router to abstain on general
# knowledge. Nothing here may contain a live Italian skill trigger or it
# becomes a poisoned sample — it would teach the router to abstain on an
# utterance a skill should own.
#
# The trigger lists were read out of the engine and the community skills,
# not guessed. Three matching rules make Italian trickier than English:
#
#  1. Built-in skills match a BAG OF WORDS, not a phrase (see `score` in
#     ari-skills/src/{current_time,date,greeting}.rs): date's
#     ["che","giorno"] fires whenever BOTH words appear anywhere. So
#     "in che giorno è nato Napoleone" is a date-skill hit, not a
#     negative. Same for ["che","data"], ["che","ora"], ["che","ore"],
#     ["ora","attuale"], ["dimmi","ora"] and greeting's ["come","va"].
#     => the standalone words `ora`/`ore`/`giorno`/`data`/`va` are banned.
#     (Note elisions split first: "un'ora" normalises to "un ora", which
#     is why no entry may elide onto a banned word either.)
#  2. `calculator.rs` matches its triggers as a SUBSTRING
#     (`input.contains`), and its list is a union of English + Italian —
#     so the English trigger `eval` fires inside "medievale", and `solve`
#     inside "risolve". Both are banned as substrings.
#  3. Community SKILL.it.md patterns are regex/whole-word over the
#     normalised text. The nasty ones are Italian polysemy: weather owns
#     \b(tempo|meteo)\b — so `tempo` meaning *duration* is poisoned —
#     search owns the bare word `trova`, which kills the otherwise
#     natural "dove si trova X", and music owns `volume`.
#
# `_it_trigger_offenders` below encodes all of that and is enforced by
# test_italian_negatives_do_not_collide_with_skill_triggers.
ITALIAN_NEGATIVE_EXAMPLES = [
    "qual è la capitale della Francia",
    "chi ha scritto Romeo e Giulietta",
    "perché il cielo è blu",
    "raccontami una barzelletta",
    "quanto dista la luna dalla terra",
    "qual è il senso della vita",
    "chi è il presidente degli Stati Uniti",
    "quanti continenti ci sono",
    "qual è la velocità della luce",
    "spiegami la fisica quantistica",
    "che cos'è la fotosintesi",
    "chi ha dipinto la Gioconda",
    "che lingua si parla in Brasile",
    "quante ossa ha il corpo umano",
    "qual è l'edificio più alto del mondo",
    "quando è stato inventato internet",
    "chi ha scoperto la gravità",
    "che cosa causa i terremoti",
    "come funziona il wifi",
    "qual è l'oceano più grande",
    "parlami dei dinosauri",
    "a che temperatura bolle l'acqua",
    "chi era Albert Einstein",
    "in che anno è finita la seconda guerra mondiale",
    "come fanno gli uccelli a volare",
    "che cos'è il DNA",
    "chi ha inventato la lampadina",
    "che cos'è il cambiamento climatico",
    "quanti pianeti ci sono nel sistema solare",
    "qual è il fiume più lungo del mondo",
    "spiegami come funziona il motore di un'automobile",
    "qual è la lingua più parlata al mondo",
    "chi è stato il primo uomo sulla luna",
    "che cosa vuol dire intelligenza artificiale",
    "come funzionano i vaccini",
    "qual è il paese più piccolo del mondo",
    "chi ha composto le quattro stagioni",
    "che cos'è la blockchain",
    "come fanno gli aerei a restare in volo",
    "che cos'è la grande muraglia cinese",
    # Chitchat and conversation closers — the Italian counterpart of the
    # English pool's tail. "ciao" is deliberately absent: it is a live
    # greeting trigger, so the farewells use arrivederci / a dopo.
    "raccontami qualcosa di interessante",
    "che libro mi consigli",
    "consigliami un film",
    "che cosa dovrei mangiare a cena",
    "mi sto annoiando",
    "ti piace la musica",
    "qual è il tuo colore preferito",
    "sei un robot",
    "che cosa sai fare",
    "grazie",
    "grazie ari",
    "lascia perdere",
    "non importa",
    "è tutto",
    "arrivederci",
    "a dopo",
    "ci vediamo dopo",
    "ho finito",
    "niente",
    "ok",
]

# Country + the definite article its name takes. Italian country names
# carry their article ("la Germania", "il Portogallo", "l'Egitto") and it
# fuses with "di" ("della"/"del"/"dell'"), so a hardcoded "della {c}"
# would emit ungrammatical Italian for over a third of this list. The
# article travels with the entity and `_it_di`/`_it_the` do the fusion.
_IT_CAPITAL_COUNTRIES = [
    ("Germania", "la"), ("Spagna", "la"), ("Francia", "la"),
    ("Grecia", "la"), ("Norvegia", "la"), ("Svezia", "la"),
    ("Polonia", "la"), ("Turchia", "la"), ("Thailandia", "la"),
    ("Croazia", "la"), ("Danimarca", "la"), ("Finlandia", "la"),
    ("Nigeria", "la"), ("Svizzera", "la"), ("Colombia", "la"),
    ("Romania", "la"), ("Scozia", "la"), ("Bulgaria", "la"),
    ("Cina", "la"), ("Russia", "la"),
    ("Irlanda", "l'"), ("Ungheria", "l'"), ("Islanda", "l'"),
    ("Austria", "l'"), ("Argentina", "l'"), ("Australia", "l'"),
    ("Indonesia", "l'"), ("India", "l'"), ("Egitto", "l'"),
    ("Etiopia", "l'"), ("Ucraina", "l'"),
    ("Portogallo", "il"), ("Brasile", "il"), ("Giappone", "il"),
    ("Canada", "il"), ("Messico", "il"), ("Perù", "il"),
    ("Vietnam", "il"), ("Marocco", "il"), ("Cile", "il"),
    ("Kenya", "il"), ("Belgio", "il"),
]

# "di" + definite article, fused. The elided form takes no space after the
# apostrophe ("dell'Egitto", never "dell' Egitto").
_IT_DI_FORMS = {"la": "della ", "il": "del ", "l'": "dell'"}


def _it_di(name: str, art: str) -> str:
    """'Germania','la' → 'della Germania'; 'Egitto',"l'" → \"dell'Egitto\"."""
    return _IT_DI_FORMS[art] + name


def _it_the(name: str, art: str) -> str:
    """'Germania','la' → 'la Germania'; 'Egitto',"l'" → \"l'Egitto\"."""
    return f"{art}{name}" if art == "l'" else f"{art} {name}"


_IT_CAPITAL_TEMPLATES = [
    "qual è la capitale {di}",
    "quale città è la capitale {di}",
    "come si chiama la capitale {di}",
    "dimmi la capitale {di}",
]

_IT_LANGUAGE_COUNTRIES = [
    "Brasile", "Svizzera", "Austria", "Belgio", "Marocco", "Nigeria",
    "Perù", "Iran", "Kazakistan", "Finlandia", "Egitto", "India",
    "Messico", "Vietnam", "Argentina", "Canada",
]

_IT_PEOPLE = [
    "Marie Curie", "Nelson Mandela", "Isaac Newton", "Leonardo da Vinci",
    "William Shakespeare", "Cleopatra", "Charles Darwin", "Gandhi",
    "Vincent van Gogh", "Nikola Tesla", "Winston Churchill", "Frida Kahlo",
    "Galileo Galilei", "Beethoven", "Aristotele", "Napoleone",
    "Stephen Hawking", "Rosa Parks", "Confucio", "Pablo Picasso",
    "Ada Lovelace", "Gengis Khan", "Florence Nightingale", "Alan Turing",
    "Dante Alighieri", "Cristoforo Colombo", "Giuseppe Garibaldi",
    "Maria Montessori",
]

# Every template is gender-neutral on purpose: "per cosa è famoso {}"
# would need famoso/famosa per entity, and the router gains nothing from
# us solving agreement here when a neutral phrasing reads just as natural.
_IT_PEOPLE_TEMPLATES = [
    "chi è {}",
    "chi era {}",
    "parlami di {}",
    "che cosa ha fatto {}",
    "raccontami la storia di {}",
]

# The article is part of the entity so the templates stay simple.
_IT_CONCEPTS = [
    "la gravità", "l'inflazione", "un buco nero", "la borsa",
    "l'evoluzione", "l'effetto serra", "il ciclo dell'acqua",
    "una recessione", "la fusione nucleare", "il big bang", "l'osmosi",
    "la democrazia", "il capitalismo", "la realtà virtuale",
    "l'entanglement quantistico", "il sistema immunitario",
    "l'apprendimento automatico", "la domanda e l'offerta",
    "l'interesse composto", "la selezione naturale", "l'inerzia",
    "la teoria della relatività",
]

_IT_CONCEPT_TEMPLATES = [
    "che cos'è {}",
    "spiegami {}",
    "puoi spiegarmi {}",
    "in che cosa consiste {}",
]

# (entity, verb) — Italian conjugates for number, so plural subjects need
# "funzionano". Carrying the verb beats restricting the list to singulars.
_IT_HOW_THINGS = [
    ("il cuore umano", "funziona"),
    ("un frigorifero", "funziona"),
    ("un forno a microonde", "funziona"),
    ("internet", "funziona"),
    ("il GPS", "funziona"),
    ("i pannelli solari", "funzionano"),
    ("una batteria", "funziona"),
    ("il cervello", "funziona"),
    ("l'energia nucleare", "funziona"),
    ("le onde radio", "funzionano"),
    ("un motore a reazione", "funziona"),
    ("la borsa", "funziona"),
    ("l'elettricità", "funziona"),
    ("l'occhio umano", "funziona"),
    ("un touchscreen", "funziona"),
    ("le cuffie con cancellazione del rumore", "funzionano"),
]

# Hand-written Italian negatives across geography, history, science and
# demographics — the less formulaic counterpart of the templated sets.
_IT_MISC_FACTUAL_NEGATIVES = [
    "qual è la montagna più alta del mondo",
    "qual è il punto più profondo dell'oceano",
    "qual è il posto più caldo della terra",
    "qual è il deserto più grande del mondo",
    "qual è la città più antica del mondo",
    "qual è il paese più popoloso del mondo",
    "in che anno è caduto il muro di Berlino",
    "quando è crollato l'impero romano",
    "quando è stata inventata la stampa",
    "in che anno l'uomo è sbarcato sulla luna",
    "quando è finita la guerra fredda",
    "che cosa ha causato la prima guerra mondiale",
    "quando si sono estinti i dinosauri",
    "quanti abitanti ha Tokyo",
    "quante persone vivono in Cina",
    "quanto è grande la Russia",
    "quanto è esteso l'oceano Pacifico",
    "qual è il simbolo chimico dell'oro",
    "quante lune ha Giove",
    "a che temperatura congela l'acqua",
    "perché sogniamo",
    "come funzionano le maree",
    "che cosa provoca l'aurora boreale",
    "perché il mare è salato",
    "quanti anni ha l'universo",
    "di che cosa è fatto il corpo umano",
    "perché le foglie cambiano colore",
    "che cos'è lo zero assoluto",
    "quante corde ha un violino",
    "qual è la valuta del Giappone",
    "chi ha costruito le piramidi",
    "perché la torre di Pisa è inclinata",
    "qual è l'animale più veloce del mondo",
    "quanto vive una tartaruga",
    "perché i vulcani eruttano",
]

# Live Italian trigger words, transcribed from the real skills:
#   ari-engine/crates/ari-skills/src/{calculator,current_time,date,
#     greeting,open,search}.rs  (the union-dictionary consts)
#   ari-skills/skills/*/SKILL.it.md  (matching.patterns)
# Whole-word entries are matched against the utterance's words, mirroring
# `contains_word` in ari-skill-loader/src/scoring.rs and the built-ins'
# `words.contains` checks. Substring entries mirror calculator.rs's
# `input.contains`.
_IT_TRIGGER_WORDS = frozenset({
    # built-ins: current_time / date / greeting / open / search
    "ora", "ore", "attuale", "giorno", "data", "oggi",
    "ciao", "salve", "buongiorno", "buonasera", "buonanotte", "va", "stai",
    "apri", "avvia", "lancia", "esegui", "cerca", "cercare", "trova",
    # weather
    "tempo", "meteo", "previsioni", "piove", "pioverà", "piovera",
    "vento", "ventoso", "uv",
    # timer / alarm / reminder
    "timer", "manca", "sveglia", "sveglie", "svegliami",
    "ricordami", "promemoria", "aggiungi", "metti",
    # music
    "riproduci", "ascolta", "pausa", "riprendi", "prossima", "successiva",
    "avanti", "salta", "precedente", "ferma", "muto", "silenzia", "volume",
    # navigation
    "portami", "indicazioni", "arrivo", "vai", "andiamo",
    # coin-flip / counter / github-zen / wasm-echo
    # wasm-echo's Italian trigger is `echo`, not `eco` — the skill name is a
    # developer term and is deliberately not translated (Keith's review).
    "moneta", "tira", "croce", "conta", "contatore", "zen", "saggezza",
    "wasm", "echo",
    # home-assistant: accendi/spegni (+ infinitives), the abbassa/alza
    # brightness family anchored to luci/luminosità, imposta/regola for
    # thermostats, apri (already banned above) / chiudi / blocca / sblocca
    # for locks and covers, attiva for scenes, and the AND-keywords
    # termostato + luci. "dove"/"dov" are banned outright rather than
    # encoding the real pattern's `dove? (e|è|sono|si trova|si trovano)`
    # adjacency — same over-conservative-single-word tradeoff as "ora"/
    # "giorno" above. "trova" is already banned; "trovano" covers the
    # plural "si trovano".
    "accendi", "accendere", "spegni", "spegnere", "imposta", "regola",
    "abbassa", "alza", "attenua", "aumenta", "riduci", "chiudi", "blocca",
    "sblocca", "luci", "luminosità", "termostato", "attiva",
    "dove", "dov", "trovano",
})

# calculator.rs matches these anywhere in the string, not as words.
# `apert`/`chius` are stems: home-assistant's status patterns match the
# participle forms (aperto/aperta/aperti/aperte, chiuso/chiusa/chiusi/
# chiuse) via a character class, so a stem substring catches all of them.
_IT_TRIGGER_SUBSTRINGS = (
    "calcola", "risolvi", "calculate", "compute", "eval", "solve",
    "apert", "chius",
)


def _it_trigger_offenders(pool: list) -> list:
    """Italian negatives that collide with a live skill trigger.

    Applies the engine's real matching rules to `pool`: whole-word for the
    union dictionaries and the SKILL.it.md keyword/regex patterns,
    substring for calculator.rs. Digits are flagged too — `has_math_content`
    only needs a digit plus any math-word substring, and Italian's "per"
    (times) hides inside perché/persone/temperatura, so a digit is enough
    to hand an utterance to the calculator."""
    offenders = []
    for n in pool:
        # Elisions split into separate words before matching (un'ora → un ora).
        words = set(n.lower().replace("'", " ").replace("’", " ").split())
        if words & _IT_TRIGGER_WORDS:
            offenders.append(n)
        elif any(s in n.lower() for s in _IT_TRIGGER_SUBSTRINGS):
            offenders.append(n)
        elif any(c.isdigit() for c in n):
            offenders.append(n)
    return offenders


def generate_italian_factual_negatives() -> list:
    """Italian counterpart of generate_factual_negatives — a phrasing-varied
    general-knowledge pool large enough (~500 uniques) for
    generate_negative_samples to scale negatives to the positive count."""
    out = []
    # Capitals + other country facts (reuse the country list).
    for tmpl in _IT_CAPITAL_TEMPLATES:
        for c, art in _IT_CAPITAL_COUNTRIES:
            out.append(tmpl.format(di=_it_di(c, art)))
    for c, art in _IT_CAPITAL_COUNTRIES:
        out.append(f"qual è la popolazione {_it_di(c, art)}")
        # "valuta", not "moneta": moneta is a live coin-flip keyword.
        out.append(f"qual è la valuta {_it_di(c, art)}")
        # "a quale continente appartiene", not "dove si trova": `trova` is
        # a live search trigger.
        out.append(f"a quale continente appartiene {_it_the(c, art)}")
    for c in _IT_LANGUAGE_COUNTRIES:
        out.append(f"che lingua si parla in {c}")
    # People + concepts — every template, every entity.
    for tmpl in _IT_PEOPLE_TEMPLATES:
        for p in _IT_PEOPLE:
            out.append(tmpl.format(p))
    for tmpl in _IT_CONCEPT_TEMPLATES:
        for concept in _IT_CONCEPTS:
            out.append(tmpl.format(concept))
    for thing, verb in _IT_HOW_THINGS:
        out.append(f"come {verb} {thing}")
        out.append(f"spiegami come {verb} {thing}")
    out.extend(_IT_MISC_FACTUAL_NEGATIVES)
    return out


def generate_factual_negatives() -> list:
    """Build a large, phrasing-varied general-knowledge negative pool.

    Every template is applied to every entity: multiple phrasings of the same
    fact ("who is X" / "who was X" / "tell me about X") all reinforce the same
    intent — general knowledge → emit no function. This multiplies the pool
    (~500 uniques) so `generate_negative_samples` can scale negatives to match
    the positive count without us hand-typing hundreds of entities."""
    out = []
    # Capitals + other country facts (reuse the country list).
    for tmpl in _CAPITAL_TEMPLATES:
        for c in _CAPITAL_COUNTRIES:
            out.append(tmpl.format(c))
    for c in _CAPITAL_COUNTRIES:
        out.append(f"what is the population of {c}")
        out.append(f"what currency is used in {c}")
        out.append(f"what continent is {c} in")
    for c in _LANGUAGE_COUNTRIES:
        out.append(f"what language do they speak in {c}")
    # People + concepts — every template, every entity.
    for tmpl in _PEOPLE_TEMPLATES:
        for p in _PEOPLE:
            out.append(tmpl.format(p))
    for tmpl in _CONCEPT_TEMPLATES:
        for concept in _CONCEPTS:
            out.append(tmpl.format(concept))
    for thing in _HOW_THINGS:
        out.append(f"how does {thing} work")
        out.append(f"explain how {thing} works")
    out.extend(_MISC_FACTUAL_NEGATIVES)
    return out


SYSTEM_PROMPT = "You are a model that can do function calling with the following functions"


# ── Build training samples ─────────────────────────────────────────────

def router_eligible_skills(skills: list) -> list:
    """Drop skills the router is never offered.

    `router_catalog()` filters on `Skill::router_eligible()`, so a skill that
    opts out (today: `search`) is never in the tools list at inference —
    training it teaches the model to call a function that isn't on the menu.
    Community skills come from manifests, which have no router_eligible
    concept, so a missing key means eligible.
    """
    return [s for s in skills if s.get("router_eligible", True)]


def assign_aliases(skills: list) -> None:
    """Set s["alias"] on each skill, mirroring ari-llm's router_alias_table:
    the router declares skills by a short alias (the final id segment) because a
    270M model can't reliably emit reverse-DNS ids. We train on the SAME names
    the runtime shows so training and inference agree. Non-unique final segments
    keep the full id to stay unambiguous."""
    from collections import Counter
    last_seg = lambda i: i.rsplit(".", 1)[-1]
    counts = Counter(last_seg(s["id"]) for s in skills)
    for s in skills:
        seg = last_seg(s["id"])
        s["alias"] = seg if counts[seg] == 1 else s["id"]


def build_tools(skills: list) -> list:
    """Convert exported skills into FunctionGemma tool declarations."""
    tools = []
    for s in skills:
        tools.append({
            "type": "function",
            "function": {
                "name": s["alias"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        })
    return tools


def build_sample(user_text, tool_calls, tools):
    """Build one training sample in the mobile-actions format."""
    messages = [
        {"role": "developer", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    if tool_calls:
        messages.append({
            "role": "assistant",
            "tool_calls": tool_calls,
        })
    else:
        messages.append({
            "role": "assistant",
            "content": (
                "I cannot assist with that request. My available tools are for "
                "specific tasks like checking the time, doing math, opening apps, "
                "searching the web, and flipping coins."
            ),
        })
    return {
        "metadata": "train",
        "tools": tools,
        "messages": messages,
    }


def generate_skill_samples(skills: list, tools: list) -> list:
    samples = []
    total_examples = 0
    for skill in skills:
        for ex in skill["examples"]:
            tool_calls = [{
                "type": "function",
                "function": {
                    "name": skill["alias"],
                    "arguments": ex["args"],
                },
            }]
            samples.append(build_sample(ex["text"], tool_calls, tools))
            total_examples += 1
    print(f"  {total_examples} skill paraphrases", file=sys.stderr)
    return samples


# Absolute floor for a corpus too small to train on at all. This is a safety
# net for the degenerate case, NOT a ratio control — see negative_target().
NEG_FLOOR = 250


# Distinguishes "the verdict iterator is exhausted" from a legitimate False
# verdict, which `next(flags, False)` could not.
_EXHAUSTED = object()


def drop_keyword_hits(all_skills: list, oracle) -> None:
    """Delete, in place, every example the keyword scorer already claims.

    `oracle` takes the flat list of example texts and returns one verdict per
    text, in order — in production that is `keyword_hits`, in tests a stub.

    The pairing between verdicts and examples is positional and entirely
    implicit, which makes it the one place in this generator that can corrupt
    the corpus SILENTLY: mis-paired verdicts drop the wrong examples and
    nothing complains. Two things guard it.

    First, the flat list is built HERE, immediately before it is consumed, from
    the same `all_skills` the walk below uses. Previously the caller flattened
    and the loop consumed, several statements apart — so reordering
    `all_skills` in between would mis-pair every example after the divergence
    while keeping the count intact, which no length check can catch. Building
    and consuming in one function makes that reorder unexpressible.

    Second, the verdict iterator must be exactly exhausted: running out
    mid-walk, or having verdicts left over at the end, both raise. That covers
    the oracle (a subprocess) returning the wrong number of verdicts.
    """
    hit_flags = oracle([e["text"] for s in all_skills for e in s["examples"]])
    flags = iter(hit_flags)
    for skill in all_skills:
        kept, dropped = [], 0
        for ex in skill["examples"]:
            verdict = next(flags, _EXHAUSTED)
            if verdict is _EXHAUSTED:
                raise ValueError(
                    f"keyword-hit verdicts ran out at skill {skill['id']!r}: got "
                    f"{len(hit_flags)} verdicts for more examples than that. The flat "
                    f"example list and all_skills have desynced — examples would be "
                    f"paired with the wrong verdicts."
                )
            if verdict:
                dropped += 1
            else:
                kept.append(ex)
        skill["examples"] = kept
        if dropped:
            print(f"    {skill['id']}: dropped {dropped}, kept {len(kept)}", file=sys.stderr)

    if next(flags, _EXHAUSTED) is not _EXHAUSTED:
        raise ValueError(
            f"keyword-hit returned {len(hit_flags)} verdicts but all_skills holds fewer "
            f"examples than that. The flat example list and all_skills have desynced — "
            f"examples were paired with the wrong verdicts."
        )


def negative_target(positive_count: int, ratio: float) -> int:
    """How many negatives to sample for `positive_count` positives.

    Abstention hit 100% on a corpus with a 1:1 positive:negative ratio, so
    that is the default and it is held explicitly. The old formula
    (`max(NEG_FLOOR, positives)`) was equivalent while positives exceeded
    250, but once the keyword-hit filter cuts positives below the floor it
    pins negatives at 250 and skews the corpus toward "emit nothing" — the
    over-abstention failure we are treating.

    NEG_FLOOR now binds only for a corpus too small to train on at all
    (< 50 positives), where an absolute floor beats a proportional one.

    This makes the floor a step function, not a smooth one: at the
    threshold, one additional positive example jumps the target down
    5x (`negative_target(49, 1.0) == 250` vs `negative_target(50, 1.0)
    == 50`). That discontinuity is intentional, not a bug — any
    absolute floor creates a step at its boundary, and a corpus with
    49 positives is too small to train on regardless of what the
    ratio-scaled value would say. Do not smooth or interpolate this;
    the floor is a degenerate-case guard, not a curve.
    """
    if positive_count < 50:
        return NEG_FLOOR
    return int(round(positive_count * ratio))


def negatives_for_locale(locale: str) -> list:
    """General-knowledge negatives for `locale`."""
    if locale == "it":
        return ITALIAN_NEGATIVE_EXAMPLES + generate_italian_factual_negatives()
    return NEGATIVE_EXAMPLES + generate_factual_negatives()


def generate_negative_samples(tools: list, target: int, pool: list) -> list:
    # De-dupe the supplied pool, then sample to the target count.
    pool = list(dict.fromkeys(pool))
    if len(pool) >= target:
        chosen = random.sample(pool, target)
    else:
        chosen = pool
        print(
            f"  WARNING: negative pool ({len(pool)}) < target ({target}); abstention "
            f"may be under-represented — add entities to generate_factual_negatives",
            file=sys.stderr,
        )
    samples = [build_sample(text, None, tools) for text in chosen]
    print(f"  {len(samples)} negative examples (target {target}, pool {len(pool)})", file=sys.stderr)
    return samples


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate the FunctionGemma dataset.")
    parser.add_argument("--locale", default="en", help="Locale to build the dataset for (default: en)")
    parser.add_argument("--filter-keyword-hits", action="store_true",
                        help="Drop examples the keyword scorer already wins. OFF by default: "
                             "measured 2026-07-19 to make the model worse in both locales. "
                             "Useful for measuring keyword-tier coverage, not for training.")
    parser.add_argument("--neg-ratio", type=float, default=1.0,
                        help="Negatives per positive (default: 1.0)")
    parser.add_argument("--augment", metavar="BANKS_DIR", default=None,
                        help="Directory of frame/slot banks (corpus/). Expands "
                             "them deterministically into the corpus — the "
                             "Google mobile-actions-scale volume recipe.")
    parser.add_argument("--allow-missing-banks", action="store_true",
                        help="With --augment: warn instead of failing when a "
                             "router-eligible skill has no frame bank.")
    parsed = parser.parse_args()
    locale, neg_ratio = parsed.locale, parsed.neg_ratio
    filter_keyword_hits = parsed.filter_keyword_hits
    augment_dir = Path(parsed.augment) if parsed.augment else None

    engine_dir = find_engine_dir()
    builtin_skills = export_skills(engine_dir, locale)
    print(f"  {len(builtin_skills)} built-in skills", file=sys.stderr)

    # Load community skills from ari-skills repo
    skills_dir = find_skills_dir()
    community_skills = load_community_skills(skills_dir, locale) if skills_dir else []
    print(f"  {len(community_skills)} community skills with examples", file=sys.stderr)

    # Merge — community skills after built-ins. Skip duplicates by id.
    builtin_ids = {s["id"] for s in builtin_skills}
    all_skills = builtin_skills + [s for s in community_skills if s["id"] not in builtin_ids]
    before = len(all_skills)
    all_skills = router_eligible_skills(all_skills)
    dropped = before - len(all_skills)
    if dropped:
        print(f"  dropped {dropped} router-ineligible skill(s) — never offered at inference",
              file=sys.stderr)
    print(f"  {len(all_skills)} total skills for training", file=sys.stderr)

    # Frame×slot augmentation — the volume half of Google's mobile-actions
    # recipe (~1,380 rows per function vs our ~20 hand-authored). Banks are
    # committed, human-reviewed JSON; the expansion is deterministic and
    # decontaminated against the held-out evals. Runs before normalisation so
    # expanded text flows through the same normalize_input path as everything
    # else, and before alias/tool assembly so counts reflect the full corpus.
    expanded_negatives = []
    if augment_dir is not None:
        from corpus_expander import expand_negatives, expand_skills, load_eval_keys
        eval_keys = load_eval_keys([
            Path(__file__).parent / "routing-eval.jsonl",
            Path(__file__).parent / "routing-eval.it.jsonl",
            # MUST stay in lockstep with the held-out guard's list below: an
            # expansion this site fails to drop is exactly what the (fatal)
            # guard then trips over, hard-failing the nightly.
            Path(__file__).parent / "routing-eval.gen.jsonl",
            Path(__file__).parent / "routing-eval.gen.it.jsonl",
        ])
        print(f"  augmenting from {augment_dir} ({locale}):", file=sys.stderr)
        extra = expand_skills(augment_dir, locale, all_skills, eval_keys,
                              allow_missing=parsed.allow_missing_banks)
        for skill in all_skills:
            skill["examples"] = skill["examples"] + extra.get(skill["id"], [])
        expanded_negatives = expand_negatives(augment_dir, locale, eval_keys)
        total = sum(len(s["examples"]) for s in all_skills)
        print(f"  corpus after augmentation: {total} positive examples",
              file=sys.stderr)

    assign_aliases(all_skills)
    tools = build_tools(all_skills)

    # OPT-IN, AND DEFAULT-OFF ON PURPOSE. Measured 2026-07-19: filtering these
    # examples out makes the model WORSE, in both locales.
    #
    #   corpus            steps   en abstain/pos   it abstain/pos
    #   unfiltered, 2ep   ~36     100% / 60%       100% / 65%
    #   filtered,   2ep   ~14      95% / 65%        86% / 65%
    #   filtered,   5ep   ~35      86% / 40%       100% / 30%
    #
    # The hypothesis was that a keyword-won example is wasted training data,
    # by analogy with the `search` fix. THAT ANALOGY IS FALSE. `search` was a
    # function never OFFERED at inference (router_eligible=false), so training
    # on it taught an impossible call. A keyword-won example is a VALID mapping
    # to an offered function: the router never sees that particular utterance
    # in production, but the example is still the model's evidence for what the
    # skill MEANS — its semantics, signature and argument shape. It is
    # redundant at inference, not useless at training. Dropping ~60% of the
    # corpus dropped most of the semantic signal for ~16 skills.
    #
    # The router being a fallback tier is a fact about INFERENCE. It does not
    # transfer to TRAINING. (It DOES transfer to EVAL — scoring the router on
    # cases it never sees measures nothing, which is why route-eval's guardrail
    # rejects them. That half is correct and stays enforced.)
    #
    # Kept behind a flag because it is still the right instrument for MEASURING
    # how much of the corpus the keyword tier owns. Do not re-enable it for
    # training without new evidence.
    if filter_keyword_hits:
        # Filter on RAW text (keyword_decision normalises internally) and
        # before the normalisation pass below, so we don't normalise text we
        # are about to discard.
        before_filter = sum(len(s["examples"]) for s in all_skills)
        drop_keyword_hits(
            all_skills,
            lambda texts: keyword_hits(engine_dir, texts, locale, skills_dir),
        )
        after_filter = sum(len(s["examples"]) for s in all_skills)
        pct = 100.0 * (before_filter - after_filter) / before_filter if before_filter else 0.0
        print(f"  keyword-hit filter: {before_filter} -> {after_filter} examples "
              f"({before_filter - after_filter} dropped, {pct:.0f}%)", file=sys.stderr)

        empty = [s["id"] for s in all_skills if not s["examples"]]
        if empty:
            sys.exit(f"ERROR: every example for {empty} is a keyword-hit — that skill "
                     f"would train with zero positives. Author oblique examples for it "
                     f"before regenerating.")
    else:
        print(f"  keyword-hit filter: OFF (default) — "
              f"{sum(len(s['examples']) for s in all_skills)} examples kept", file=sys.stderr)

    # Normalise the survivors through the engine's OWN normalize_input. The
    # router only ever sees normalize_input()'d text at inference; training on
    # raw text trains it on a distribution it is never served. Args and
    # descriptions are schema, not utterances — they are NOT normalised.
    flat = [ex for s in all_skills for ex in s["examples"]]
    for ex, norm in zip(flat, normalize_texts(engine_dir, [e["text"] for e in flat], locale)):
        ex["text"] = norm
    print(f"  normalised {len(flat)} skill paraphrases ({locale})", file=sys.stderr)

    # HELD-OUT GUARD, unconditional. The routing evals are the promotion
    # gate's yardstick; a training row that matches an eval case converts the
    # gate into a memorisation test. The corpus expander decontaminates its
    # own output, but hand-written sources (manifest examples, the negative
    # pools below) historically leaked — 2026-07-19 found two live eval NONE
    # cases sitting in the hand-written EN negative pool. Positives that
    # collide are a hard error (someone must consciously change one side);
    # colliding negatives are dropped and reported.
    from corpus_expander import load_eval_keys, loose_key
    heldout_keys = load_eval_keys([
        Path(__file__).parent / "routing-eval.jsonl",
        Path(__file__).parent / "routing-eval.it.jsonl",
        # Generated eval banks (generate-eval.py) are held out exactly like
        # the hand-written spine — a training row matching one would turn
        # its derive_floor.py numbers into a memorisation test.
        # load_eval_keys skips files that don't exist yet.
        Path(__file__).parent / "routing-eval.gen.jsonl",
        Path(__file__).parent / "routing-eval.gen.it.jsonl",
    ])
    poisoned = [ex["text"] for ex in flat if loose_key(ex["text"]) in heldout_keys]
    if poisoned:
        sys.exit(f"ERROR: skill example(s) collide with held-out eval cases: "
                 f"{poisoned[:5]} — change the example or the eval case, "
                 f"deliberately, not both silently.")

    print("Generating samples:", file=sys.stderr)
    skill_samples = generate_skill_samples(all_skills, tools)
    # Scale negatives to the positive count so "emit nothing" stays well
    # represented as skills are added. (We deliberately do NOT mix in Google's
    # mobile-actions demo dataset: the base model is already function-calling
    # pretrained, those are actions Ari doesn't support, and 9,654 always-call
    # samples drown abstention — the root of the r70→r75 regression.)
    neg_target = negative_target(len(skill_samples), neg_ratio)
    print(f"  negative target {neg_target} for {len(skill_samples)} positives "
          f"(ratio {neg_ratio})", file=sys.stderr)
    negative_pool = normalize_texts(
        engine_dir, negatives_for_locale(locale) + expanded_negatives, locale)
    before_guard = len(negative_pool)
    negative_pool = [t for t in negative_pool if loose_key(t) not in heldout_keys]
    if before_guard != len(negative_pool):
        print(f"  held-out guard: dropped {before_guard - len(negative_pool)} "
              f"negative(s) colliding with eval cases", file=sys.stderr)
    print(f"  normalised {len(negative_pool)} negatives ({locale})", file=sys.stderr)
    negative_samples = generate_negative_samples(tools, neg_target, negative_pool)

    all_samples = skill_samples + negative_samples
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.9)
    train = all_samples[:split_idx]
    eval_ = all_samples[split_idx:]

    for s in train:
        s["metadata"] = "train"
    for s in eval_:
        s["metadata"] = "eval"

    combined = train + eval_
    print(
        f"\nTotal: {len(combined)} samples ({len(train)} train, {len(eval_)} eval)",
        file=sys.stderr,
    )

    for sample in combined:
        print(json.dumps(sample, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
