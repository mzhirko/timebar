# Rule packs: teach the tool any time limit

## First, what does the tool do *without* any rule pack?

Almost everything. The timeline, the disputed-dates detection, the
what-if recomputation, the interval questions, the contradiction
report — none of it needs a rule pack. Rule packs power exactly one
feature: **the deadline check** ("was this done within the legal time
limit, and show the working").

## What is a rule pack?

One legal time limit, written down as a small folder of data files
instead of program code. The tool ships with two:

| Pack | Rule | Why it's included |
|---|---|---|
| `uk/era-1996-s111` | UK unfair dismissal: claim within **3 months beginning with** the termination date | A real statute (openly licensed), showing *inclusive* day-counting |
| `example/appeal-21-days` | Invented rule: appeal within **21 days from** the decision | Fully synthetic template to copy, showing *exclusive* day-counting |

Two packs on purpose: they use opposite counting conventions
("beginning with" = first day counts, deadline one day earlier;
"from the date" = it doesn't), and the tool reads the right convention
**from each statute's own wording** — the same program code handles
both, unchanged. Any number of packs can sit side by side; you choose
one per check with `--rule`.

## Write your own in five steps

Suppose your rule is: *"A complaint must be lodged within 6 weeks from
the date of notification."*

**1. Copy the template**

```bash
cp -r rulepacks/example/appeal-21-days rulepacks/my/complaint-6-weeks
```

**2. Put your rule into `statute.tdg.json`.** Open it — it is short.
You change three things:

- the sentence (`source_text`, and each fact's `sentence`) — paste the
  clause's actual wording; the tool reads the period ("6 weeks") and
  the counting convention ("from the date") from this text;
- the two facts: the event the clock starts from
  (`"date of notification"`, role `START`) and the thing that must
  happen (`"lodging of the complaint"`, role `END`);
- the connection between them: `"constraint_expr": "6 weeks"`.

**3. List your word-equivalents in `aliases.json`.** How do real
documents refer to the start event? `"notification"`, `"the notice"`,
`"date of service"` — whatever your documents actually say. This only
helps the tool match wording; it never changes the arithmetic.

**4. Write two or three test cases with known answers.** Small
invented cases in `gold_cases/` (one in time, one too late, one with
missing information) and the correct answers in `expected.json`.
Invented means invented — see `DATA_POLICY.md`; never paste a real
case.

**5. Validate until it passes.**

```bash
tdg rulepack validate rulepacks/my/complaint-6-weeks
```

This checks the files, confirms the rule is readable from your
wording, runs your test cases, and prints exactly what the tool
understood — including which counting convention it read — so you can
verify it read your rule the way you meant it. When it prints `PASS`,
use it:

```bash
tdg-chrono deadline ./my-case --rule rulepacks/my/complaint-6-weeks/statute.tdg.json --explain
```

## What can a pack express today, honestly?

One clause of the form "*X must happen within PERIOD of EVENT*", with
calendar-exact periods (days/weeks/months/years), either counting
convention, and an optional clock-pause (conciliation-style) supplied
per case. English wording only, for now. Multi-clause interactions,
discretionary extensions ("such further period as is reasonable"), and
non-English statutes are not expressible yet — a pack cannot smuggle
them in, which is the point: what the tool applies is only ever what
you can read in the pack.
