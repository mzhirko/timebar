# Rule pack: UK contract limitation period (Limitation Act 1980, s.5)

The rule this pack encodes: an action founded on simple contract may
not be brought more than **six years from the date the cause of action
accrued**. The wording "from the date" means the accrual day itself
does not count (exclusive counting) — compare the s.111 pack, whose
"beginning with" means the opposite. The engine reads the convention
from the wording; run `tdg rulepack validate rulepacks/uk/limitation-act-1980-s5`
to see what it understood.

Usage:

```bash
tdg-chrono deadline ./my-case --rule rulepacks/uk/limitation-act-1980-s5/statute.tdg.json --explain
```

Files: `statute.tdg.json` (the clause and its machine-readable
summary), `aliases.json` (word-equivalents, e.g. "breach of contract"
for the accrual date), `gold_cases/` + `expected.json` (three invented
test cases with known answers). To write your own pack, see
`rulepacks/README.md`.

Provenance, per `DATA_POLICY.md`: statute wording paraphrases the
Limitation Act 1980 s.5 from legislation.gov.uk, Open Government
Licence v3.0 ("Contains public sector information licensed under the
Open Government Licence v3.0"). All test cases are synthetic — every
name, date and document is invented.

Caveat: s.5 states the basic period only. Postponement and extension
rules elsewhere in the Act (disability, fraud, concealment,
acknowledgment) are not encoded and are outside what a pack can
currently express.
