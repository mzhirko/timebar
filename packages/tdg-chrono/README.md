# tdg-chrono

The user-facing tool: a folder of case documents in, a sourced
chronology out. Built on `tdg-core`.

```bash
pipx install tdg-chrono
tdg-chrono build ./bundle -o ./out --from-tdgs     # offline, no model
tdg-chrono view ./my-case                          # point-and-click viewer
```

What you get: an Excel/CSV/JSON timeline with one row per real-world
event, every row carrying the quotes it came from, disagreements
between documents shown as highlighted rows (never silently resolved),
undatable items listed rather than dropped.

Commands: `build` (documents to timeline) · `view` (browser UI) ·
`correct` (reversible fixes that survive re-runs) · `deadline`
(check the bundle against a statute's time limit, working shown) ·
`interval` ("was X live on this date?") · `whatif` ("if this date
moves, what else moves?") · `contradictions` (where documents disagree).

Extras: `[llm]` LLM extractor (works with local Ollama) · `[nlp]`
offline HeidelTime/spaCy extractor · `[pdf]` PDF intake · `[viewer]`
the browser UI. With no extras installed, everything works on
pre-extracted TDG files and nothing ever leaves your machine.

Extractors are plugins: implement one `extract()` method, register a
`tdg.extractors` entry point, and `--extractor yourname` works.

Full documentation: repository README. Not legal advice — see
DISCLAIMER.md.
