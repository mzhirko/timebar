# Rule pack: 21-day appeal limit — SYNTHETIC EXAMPLE

**This rule is invented** (category 2 of `DATA_POLICY.md`): it belongs
to no real jurisdiction, and every test case in `gold_cases/` is made
up. It exists as the copy-me template for writing your own rule pack —
see `rulepacks/README.md` for the step-by-step guide.

The rule it encodes: *"An appeal must be filed within 21 days from the
date of the decision."* Note the wording "from the date": the tool
reads this as *exclusive* counting (the decision day itself doesn't
count), unlike the UK pack's "beginning with" (*inclusive*). Same
program code, opposite conventions, decided by the statute's own words
— run `tdg rulepack validate rulepacks/example/appeal-21-days` to see
it spelled out.
