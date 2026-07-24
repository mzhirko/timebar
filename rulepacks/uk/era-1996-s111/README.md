# Rule pack: UK unfair dismissal time limit (Employment Rights Act 1996, s.111)

## What is this folder?

One rule pack of many — the tool is not limited to this statute. Any
time limit can be added as a folder like this one without touching the
program: see `rulepacks/README.md` for the guide and
`rulepacks/example/appeal-21-days/` for a synthetic template to copy.
This pack is the shipped *real-statute* example.

UK law says that an unfair dismissal claim must reach the employment
tribunal **within three months of the dismissal**. This folder teaches
Timebar that rule — not by putting it in the program's code, but as a
small set of data files the program reads.

That distinction matters for trust: the tool's *code* contains no legal
rules at all. Every rule it applies lives in a folder like this one,
where you can open it, read it, and check it against the actual
statute. To support a different rule — or a different country — you add
a folder like this; the program itself never changes. You can prove
that claim yourself:

```bash
tdg rulepack validate rulepacks/uk/era-1996-s111
```

which checks this folder end-to-end and prints
`PASS — pack usable with zero engine changes`.

## How do I use it?

Point a deadline check at it:

```bash
tdg-chrono deadline ./my-case --rule rulepacks/uk/era-1996-s111/statute.tdg.json --explain
```

The tool finds the dismissal date in your documents, applies the
three-month limit, and prints every step of the calculation — including
which sentence of the statute each number came from.

## What's in the files?

- **`statute.tdg.json`** — the rule itself: the statute's sentence
  ("…within the period of three months beginning with the effective
  date of termination…") plus a machine-readable summary of it (which
  event the clock starts from, what must happen, how long the period
  is). The tool reads the fine print from the sentence's own wording —
  for example, "beginning with" means the first day counts, which
  makes the deadline one day earlier than naive counting. Nothing is
  hidden in code; the sentence you can read is the sentence the tool
  uses.
- **`gold_cases/`** — three invented test cases: one claim filed in
  time, one filed too late, and one where the documents don't contain
  enough information to answer. They exist so the pack can be checked
  automatically.
- **`expected.json`** — the correct answer for each test case. The
  `validate` command runs every test case and compares.
- **`aliases.json`** — a translation list between the statute's formal
  wording and how real documents talk. The statute says "effective
  date of termination"; a judgment might just say "the dismissal".
  This file lists such equivalents so the tool can match them. It only
  helps with *matching words* — it never changes the rule or the
  arithmetic — and it loads automatically when you use this pack.
  (The tool ships with no such lists built in; each rule pack brings
  its own. Every result records which lists were active, so a
  calculation can always be reproduced exactly.)

## Where does the text come from? May I reuse it?

Per the repository's `DATA_POLICY.md`:

- The statute wording paraphrases ERA 1996 s.111 from legislation.gov.uk,
  used under the Open Government Licence v3.0. Attribution: "Contains
  public sector information licensed under the Open Government Licence
  v3.0."
- The test cases are **synthetic examples** — every name, date and
  document is invented. No real case is described or implied.

## Details for careful readers

- Periods are computed with real calendar months and leap years — "3
  months" is never approximated as 90 days.
- Whether the first day counts is **read from the statute's wording**,
  not assumed: "beginning with" means it counts (deadline one day
  earlier); "from the date" means it doesn't. The calculation printout
  always says which reading was used and quotes the phrase.
- UK "early conciliation" (s.207B), which can pause the clock, is a
  general mechanism in the tool: it activates only when your case
  supplies the conciliation start and end dates. It is not part of
  this pack.
- For thesis/paper reproduction: published evaluation numbers were
  produced with this pack's `aliases.json` loaded. That happens
  automatically whenever this pack is used, and the derivation trace
  records it (`vocabulary_sources`).

## Making your own rule pack

Copy this folder's structure: write the statute's clause into
`statute.tdg.json`, add your jurisdiction's word-equivalents to
`aliases.json`, write a few test cases with known answers, and run
`tdg rulepack validate` on it until it passes. No program code needs
to change.
