# Timebar

Timebar builds a chronology from a set of legal documents and checks
dates against statutory time limits.

Given a folder of documents from one case (letters, claim forms,
responses, contracts), it:

1. extracts the dated facts from each document, keeping the sentence
   each fact came from;
2. identifies facts in different documents that refer to the same
   event;
3. outputs a timeline with one row per event, sorted by date.

Each row lists its date, a label, the source documents, the source
sentences, and a status: `agreed` (multiple documents, same date),
`disputed` (documents give different dates; all values are shown),
`single source`, or `derived` (computed from another date, e.g.
"28 days after service"; the calculation is shown). Facts without a
usable date are listed in a separate section instead of being dropped.

It can also check the documents against a legal deadline (for example
"a claim must be presented within three months of dismissal") and
print the calculation step by step.

This is not legal advice. Extraction can be wrong or incomplete;
output must be checked against the quoted sources. See
[DISCLAIMER.md](DISCLAIMER.md).

## Installation

Requires Python 3.10+.

```bash
pipx install tdg-chrono
```

Optional extras:

```bash
pip install 'tdg-chrono[viewer]'   # browser interface
pip install 'tdg-chrono[llm]'      # LLM-based extractor (OpenAI-compatible API, incl. local Ollama)
pip install 'tdg-chrono[nlp]'      # offline extractor (HeidelTime + spaCy)
pip install 'tdg-chrono[pdf]'      # PDF input
```

Without extras, the tool works on pre-extracted input files and makes
no network connections.

## Usage

### Build a timeline

```bash
tdg-chrono build ./my-bundle -o ./out --extractor llm   # from raw documents (.pdf/.docx/.txt)
tdg-chrono build ./my-bundle -o ./out --from-tdgs       # from already-extracted files
```

Output: `out/chronology.xlsx`, `out/chronology.csv`,
`out/chronology.json`. To try it without any setup, the repository
includes a fabricated example case:

```bash
tdg-chrono build examples/sample-bundle -o ./out --from-tdgs
```

which produces five events, including one disputed date and one
derived deadline (see `examples/sample-bundle/README.md` for the
expected output).

### Browser interface

```bash
tdg-chrono view ./my-case
```

Opens a local web page. Upload documents, inspect each timeline row
and its source sentences, edit dates, remove or confirm rows, run
deadline checks, and download the results. Requires the `[viewer]`
extra.

### Correct extraction mistakes

```bash
tdg-chrono correct corrections.json add --op edit-date --doc et1 --fact f1 --date 2025-07-12
tdg-chrono correct corrections.json add --op reject   --doc response --fact f9
tdg-chrono build ./my-bundle -o ./out --corrections corrections.json
```

Operations: `accept`, `reject`, `edit-date`, `edit-label`, `merge`,
`split`. Corrections are stored in the corrections file and applied on
every rebuild. Source documents are not modified; deleting an entry
from the file undoes the correction. The viewer writes the same file.

### Check a deadline

```bash
tdg-chrono deadline ./my-bundle --rule rulepacks/uk/era-1996-s111/statute.tdg.json --explain
```

Prints the applicable period, the sentence of the statute it was read
from, the anchor date found in the documents, whether the first day
counts (determined from the statute's wording), the arithmetic, and
the result as a date comparison. If required information is missing,
it reports what is missing instead of guessing.

Time limits are defined in "rule packs": data folders containing the
statute clause, vocabulary, and test cases. Two are included (the UK
unfair-dismissal limit and a fabricated 21-day appeal rule). To add
one, see [rulepacks/README.md](rulepacks/README.md). The timeline
commands do not use rule packs.

### Other commands

```bash
tdg-chrono whatif ./my-bundle --set contract:f1=2025-08-01     # recompute dates that depend on a moved date
tdg-chrono interval ./my-bundle --doc contract --entity "non-compete" --on 2025-06-01   # was it in force on a date
tdg-chrono interval ./my-bundle --between letter:f1 et1:f3     # order of two events
tdg-chrono contradictions ./my-bundle                          # list conflicting statements
tdg validate ./my-bundle                                       # check input files against the format schema
tdg rulepack validate rulepacks/uk/era-1996-s111               # check a rule pack
```

## Docker

```bash
docker build -t timebar .
docker run --rm -v "$PWD/mycase:/case" timebar build /case/tdgs -o /case/out --from-tdgs
docker run --rm -p 8501:8501 -v "$PWD/mycase:/case" timebar view /case
```

`docker-compose.yml` runs the tool together with Ollama for LLM
extraction without external services.

## Limitations

- Documents must be in English. The wording that carries counting
  rules ("within three months", "beginning with") is matched with
  English patterns.
- Rule packs express single-clause limits of the form "X must happen
  within PERIOD of EVENT". Multi-clause interactions and discretionary
  extensions are not supported.
- Extraction quality depends on the extractor used and requires
  review. The `correct` command and the viewer exist for this reason.

## Repository layout

```
packages/tdg-core/      file format (JSON schema), date arithmetic engine, cross-document linking
packages/tdg-chrono/    command-line tool and viewer
rulepacks/              time limits as data files, with an authoring guide
examples/sample-bundle/ fabricated example case
```

`tdg-core` is a separate package with no LLM or NLP dependencies; the
research release (benchmarks and evaluation code, published
separately) depends on the same published version of it.

Input and interchange use the TDG format: one JSON file per document,
listing dated facts with text positions and the constraints between
them. Schema: `packages/tdg-core/src/tdg_core/schema/tdg-v1.json`.
Extractors are plugins registered via the `tdg.extractors` entry
point; any program that emits schema-valid JSON can also be used
without installing anything from this repository.

## Data and licensing

Code: Apache-2.0. UK statutory wording from legislation.gov.uk under
the Open Government Licence v3.0. All example documents and test cases
are fabricated; see [DATA_POLICY.md](DATA_POLICY.md) for the rules on
what data may be added to this repository.

"Timebar" is a working title; package names are `tdg-*`.
