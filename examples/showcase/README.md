# Showcase bundle — SYNTHETIC EXAMPLE

All names, dates, documents and events in this folder are invented
(category 2 of DATA_POLICY.md); no real case, party or filing is
described or implied. Four documents: three from one fabricated matter
(letter, claim form, response) and one from an unrelated second matter
(okonkwo_dismissal), included to demonstrate matter separation: the
linker must not connect events across the two cases.

    tdg-chrono build examples/showcase/tdgs -o ./out --from-tdgs

Expected: 9 events (1 disputed, 1 unplaced), with no cross-matter rows.
