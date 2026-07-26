# Timebar

Timebar turns a folder of case documents into a chronology you can check,
and measures that chronology against statutory time limits.

It is built for the job a paralegal does by hand: read every document in a
bundle, write down what happened when, notice where two documents disagree,
and work out whether a claim was brought in time.

## What it does

Given a folder of documents from one case (letters, claim forms, responses,
contracts), Timebar:

1. **Extracts the dated facts** from each document, keeping the exact
   sentence each fact came from.
2. **Links facts across documents** that describe the same real-world event.
3. **Writes one timeline**, one row per event, sorted by date.

Every row carries its date, a label, the source documents, the source
sentences, and a status:

| status | meaning |
|---|---|
| `agreed` | two or more documents give the same date |
| `disputed` | documents disagree; every value and every quote is shown |
| `single source` | only one document mentions it |
| `derived` | computed from another date, with the working shown |

Facts with no usable date are listed in their own section rather than
dropped.

Two design rules follow from the domain:

**Nothing is silently resolved.** When documents conflict, the row shows
both values and both quotes. The tool never picks a winner, because which
date controls is a legal question.

**The model only reads; the code decides.** An LLM is used for extraction
and nothing else. Every comparison, every date calculation and every
deadline is deterministic code with a schema and tests behind it. That is
why `tdg-core` installs with no AI dependencies and its test suite runs
offline.

### Deadline checking

Timebar can check a bundle against a statutory time limit, for example "a
claim must be presented within three months beginning with the effective
date of termination", and print the whole calculation:

```
Rule (era1996_s111):
  presentation of the complaint = effective date of termination + 3m - 1 day
Period: 3 month(s) - 1 day
  anchor-day counting: DISCOVERED from the statute's own wording
Anchor: effective date of termination = 2025-07-12  [dismissal_letter:f1]
  quote: "Your employment terminates with effect from 12 July 2025."
Deadline: 2025-07-12 + 3 month(s) - 1 day = 2025-10-11
Result: the action (2025-10-01) falls 10 day(s) before the deadline.
```

It gives a derivation, not a verdict, and names what is missing rather than
guessing when it cannot compute an answer.

This is not legal advice. Extraction can be wrong or incomplete, and output
must be checked against the quoted sources. See [DISCLAIMER.md](DISCLAIMER.md).

## Install

Requires Python 3.10 or later.

From a clone of this repository:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ./packages/tdg-core -e ./packages/tdg-chrono
```

Optional extras, added in the same way:

```bash
pip install -e './packages/tdg-chrono[viewer]'   # browser interface
pip install -e './packages/tdg-chrono[llm]'      # LLM extractor, any OpenAI-compatible API
pip install -e './packages/tdg-chrono[pdf]'      # PDF input
pip install -e './packages/tdg-chrono[nlp]'      # offline extractor (HeidelTime and spaCy)
```

Check the install:

```bash
python -m pytest packages -q
tdg-chrono build examples/sample-bundle -o ./out --from-tdgs
```

The example bundle is fabricated and produces five events, one of them a
disputed termination date and one a derived deadline.

Without extras the tool works on pre-extracted input and makes no network
connections at all.

## Set up local models

Timebar talks to any OpenAI-compatible endpoint, so a local
[Ollama](https://ollama.com) install keeps documents on your machine.

```bash
ollama pull llama3            # extraction
ollama pull nomic-embed-text  # optional, for matching paraphrased event names
```

Point Timebar at it:

```bash
export OPENAI_API_KEY=ollama                       # any non-empty value
tdg-chrono build ./my-bundle -o ./out \
  --extractor llm --model llama3 --base-url http://localhost:11434/v1 \
  --embed-model nomic-embed-text --embed-base-url http://localhost:11434/v1
```

A hosted API works the same way, without `--base-url`:

```bash
export OPENAI_API_KEY=sk-...
tdg-chrono build ./my-bundle -o ./out --extractor llm --model gpt-4o-mini
```

There is no default model. `--model`, or the `TDG_LLM_MODEL` environment
variable, is required. If extraction finds no dated facts at all, the build
exits with code 3 rather than writing an empty chronology, because that
almost always means a misconfigured model rather than a bundle with no
dates. `--allow-empty` overrides this.

## Usage

### Build a timeline

```bash
tdg-chrono build ./my-bundle -o ./out --extractor llm --model llama3 \
    --base-url http://localhost:11434/v1
tdg-chrono build ./my-bundle -o ./out --from-tdgs   # from already-extracted files
```

Writes `out/chronology.xlsx`, `out/chronology.csv` and
`out/chronology.json`.

Every build also runs a **recall audit**: a regex sweep of the original
document text that reports any date no extracted fact accounts for. This
catches the extractor's silent failure mode, where a date sits in the source
but never becomes a fact and the chronology still looks complete. Its
warnings are gaps in extraction, not errors in the timeline. It finds
explicit dates only, so silence is not proof of completeness.

### Browser interface

```bash
tdg-chrono view ./my-case
```

Opens a local page where you can upload documents, read each row against its
source sentences, edit or remove rows, run deadline checks and download the
results. Needs the `[viewer]` extra.

### Correct mistakes

```bash
tdg-chrono correct corrections.json add --op edit-date --doc et1 --fact f1 --date 2025-07-12
tdg-chrono correct corrections.json add --op reject --doc response --fact f9
tdg-chrono build ./my-bundle -o ./out --corrections corrections.json
```

Operations are `accept`, `reject`, `edit-date`, `edit-label`, `merge` and
`split`. Corrections live in their own file and are re-applied on every
rebuild. Source documents are never modified, and deleting an entry undoes
the correction. The viewer writes the same file.

### Check a deadline

```bash
tdg-chrono deadline ./my-bundle \
    --rule rulepacks/uk/era-1996-s111/statute.tdg.json --explain
```

Time limits are defined in **rule packs**: data folders holding the statute
clause, its vocabulary and gold test cases. Three are included. Adding one
needs no code change, and `tdg rulepack validate` checks a pack against its
own test cases. See [rulepacks/README.md](rulepacks/README.md).

Where a statute does not count some period against its own limit, pass that
period and the engine applies it:

```bash
tdg-chrono deadline ./my-bundle --rule .../statute.tdg.json \
    --tolled-from 2025-08-01 --tolled-to 2025-08-21
```

The engine implements only the shape of that rule: a start, an end, and an
optional floor after the end. What qualifies, what it is called and whether
a floor applies are declared by the rule pack, so no jurisdiction's version
is written into the engine.

### Ask questions about a bundle

```bash
tdg-chrono ask ./my-case "when did the employment end?" \
    --model llama3 --base-url http://localhost:11434/v1
```

This is retrieval-augmented generation with the timeline consulted first.
The established dates go into the prompt before any prose, then the relevant
sentences are retrieved, then the model answers with quotes. Ordering is the
point: a model that has read the computed dates before it reads the
documents is far less likely to pull a date out of a sentence and do its own
arithmetic on it.

What that buys, on a real bundle:

- **Disagreements survive.** Asked when employment ended, it answers *"the
  documents disagree"* and gives both dates. Plain retrieval would hand the
  model whichever passage ranked first.
- **Calculated dates are available.** A response deadline stated nowhere in
  the documents is computed by the engine and can be answered.
- **Cases stay apart.** Where a folder holds more than one matter, the facts
  are grouped by case and the model is told not to combine them.
- **The answer is checked.** Every date in the reply is compared against the
  established facts; any the model invented or mis-copied is reported:

```
WARNING  the answer states date(s) that were never established from these
         documents:
           12 July 2025
         The model either did its own arithmetic or slipped.
```

`--show-prompt` prints exactly what the model was given. `--json` returns
the answer with every fact and passage it rested on, including character
offsets so a citation points at the sentence.

For the context block alone, without a model writing anything:

```bash
tdg-chrono context ./my-case --about "termination"
```

And to find out which stored answers a correction invalidates:

```bash
tdg-chrono stale ./my-case --changed et1:f5
```

## Choosing models

The tool talks to a language model for three unrelated jobs. They need not
be the same model, or the same provider, and each is configured separately.

| role | what it does | flag | environment |
|---|---|---|---|
| **extract** | reads documents into dated facts | `--model` on `build` | `TDG_EXTRACT_MODEL` |
| **answer** | writes the prose in `ask` | `--model` on `ask` | `TDG_ANSWER_MODEL` |
| **embed** | matches paraphrased names, ranks passages | `--embed-model` | `TDG_EMBED_MODEL` |

Each role resolves in this order: the command-line flag, then its own
environment variable, then a shared fallback for the common case of one
provider for everything.

| setting | per-role variable | shared fallback |
|---|---|---|
| model | `TDG_{ROLE}_MODEL` | `TDG_LLM_MODEL` (extract and answer) |
| endpoint | `TDG_{ROLE}_BASE_URL` | `OPENAI_BASE_URL` |
| key | `TDG_{ROLE}_API_KEY` | `OPENAI_API_KEY` |

**One provider for everything** — the simple case:

```bash
export TDG_LLM_MODEL=llama3
export OPENAI_BASE_URL=http://localhost:11434/v1
export TDG_EMBED_MODEL=nomic-embed-text
```

**A different provider per role** — a small local model reading documents,
a larger hosted one writing answers, a dedicated embedding service:

```bash
export TDG_EXTRACT_MODEL=llama3
export TDG_EXTRACT_BASE_URL=http://localhost:11434/v1

export TDG_ANSWER_MODEL=gpt-4o-mini
export TDG_ANSWER_API_KEY=sk-...

export TDG_EMBED_MODEL=nomic-embed-text
export TDG_EMBED_BASE_URL=http://localhost:11434/v1
```

The keys stay independent, so a hosted answering model and a local extractor
do not fight over one credential. Any OpenAI-compatible endpoint works.

A role with nothing configured is simply switched off: no embedder means
lexical matching, which is a supported way to run. Extraction is the
exception, since it cannot proceed without a model and says so:

```
error: no model configured for answer.
       Pass --model, or set TDG_ANSWER_MODEL.
```

`build` prints which models it is using, so a mixed setup is visible in the
run rather than only in your shell.

### Other commands

```bash
tdg-chrono whatif ./my-bundle --set contract:f1=2025-08-01
tdg-chrono interval ./my-bundle --between letter:f1 et1:f3
tdg-chrono interval ./my-bundle --doc contract --entity "non-compete" --on 2025-06-01
tdg-chrono contradictions ./my-bundle
tdg-chrono stale ./my-bundle --changed et1:f5
tdg validate ./my-bundle
tdg rulepack validate rulepacks/uk/era-1996-s111
```

## Configuration

### Keeping cases apart

One bundle should hold one case. When a folder mixes matters, documents
about different clients can be merged into a single row, which invents a
disagreement between people who have never met.

Timebar never guesses which documents belong together. It uses what the
documents declare:

- a `matter` field on a document (name it with `--matter-field`), or
- the `parties` named in the document, which the LLM extractor fills in.

Documents declaring different matters, or naming no party in common, are
never linked. When neither is present, everything links freely and the run
says so.

### Document intake

Documents reach the extractor as raw text with whitespace normalised, and
nothing is discarded. `--clean` opts into an aggressive cleaner that strips
headers, footers and two-column artifacts. It is built for noisy PDF corpora
and will remove body text from ordinary correspondence, so it is off by
default; when on, every span it removes is printed.

### Options reference

| option | effect |
|---|---|
| `--extractor` | registered extractor name, default `llm` |
| `--model`, `--base-url` | extraction model and endpoint (`TDG_LLM_MODEL`, `OPENAI_BASE_URL`) |
| `--embed-model`, `--embed-base-url` | optional embedder for entity names (`TDG_EMBED_MODEL`, `TDG_EMBED_BASE_URL`) |
| `--matter-field` | which key carries the matter identifier, default `matter` |
| `--linking` | `composed` (default) or `gated`, the older stricter matcher |
| `--clean` | run the aggressive text cleaner first |
| `--allow-empty` | exit 0 even when extraction finds nothing |
| `--corrections` | corrections file to re-apply |
| `--formats` | any of `xlsx`, `csv`, `json`, `docx` |

## Docker

```bash
docker build -t timebar .
docker run --rm -v "$PWD/mycase:/case" timebar build /case/tdgs -o /case/out --from-tdgs
docker run --rm -p 8501:8501 -v "$PWD/mycase:/case" timebar view /case
```

`docker-compose.yml` runs Timebar alongside Ollama, so LLM extraction works
without any external service.

## Limitations

- **English only.** The wording that carries counting rules ("within three
  months", "beginning with") is matched with English patterns.
- **Single-clause time limits.** Rule packs express limits of the form "X
  must happen within PERIOD of EVENT". Multi-clause interactions and
  discretionary extensions are not supported.
- **Output varies between runs.** LLM extraction is not deterministic, so
  the same bundle and the same model can produce different timelines from
  one run to the next. In testing, three runs of one three-document bundle
  found 0, 0 and 1 disputes, and one run failed a document outright on
  malformed model output. Treat a single run as one reading of the bundle,
  not as the answer. The recall audit reports what a run missed, and it is
  worth reading every time.
- **Extraction quality needs review regardless.** The `correct` command and
  the viewer exist for this reason, and every row carries the sentence it
  came from so it can be checked.
- **Relative expressions need a stated relationship.** "Within 28 days of
  service" is placed on the timeline only when the extractor also records
  what it counts from.

## Repository layout

```
packages/tdg-core/       file format (JSON schema), date arithmetic, cross-document linking
packages/tdg-chrono/     command-line tool, extractors, viewer
rulepacks/               time limits as data, with an authoring guide
examples/sample-bundle/  fabricated example case
```

`tdg-core` is a separate package with no LLM or NLP dependencies.

Input and interchange use the TDG format: one JSON file per document listing
dated facts with text positions and the constraints between them. The schema
is at `packages/tdg-core/src/tdg_core/schema/tdg-v1.json`. Extractors are
plugins registered through the `tdg.extractors` entry point, so any program
that emits schema-valid JSON can be used without installing anything from
this repository.

## Data and licensing

Code is Apache-2.0. UK statutory wording comes from legislation.gov.uk under
the Open Government Licence v3.0. All example documents and test cases are
fabricated. [DATA_POLICY.md](DATA_POLICY.md) sets out what data may be added
to this repository.

"Timebar" is a working title; the packages are named `tdg-*`.
