# Sample case — SYNTHETIC EXAMPLE

An invented employment dispute, included so you can try the tool in
one command with nothing to configure. **All names, dates, documents
and events are made up** (category 2 of `DATA_POLICY.md`); no real
case, party or filing is described or implied.

The three JSON files play the roles of a dismissal letter, a tribunal
claim form, and the employer's response. They are already in the
tool's extracted-facts format (called TDG — each dated fact with the
sentence it came from), which is why no extractor or model is needed:

    tdg-chrono build examples/sample-bundle -o ./out --from-tdgs

What you should see: **5 events** on the timeline —

- one **agreed** (both sides state the same employment start date),
- one **DISPUTED** (the letter says the dismissal took effect on
  12 July, the claim form says 14 July — the row shows both, with
  both quotes),
- two **single-source** (mentioned by one document only),
- one **derived** (the response deadline, computed as "28 days after
  service" — no document states the date directly),

plus one **unplaced** item (a sentence with no date in it, listed
rather than deleted).

The same folder also works with `tdg-chrono view`, and with the
`deadline`, `whatif`, `interval` and `contradictions` commands — it is
the bundle used throughout the main README's examples.
