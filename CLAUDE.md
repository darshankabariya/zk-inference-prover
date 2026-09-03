# Rules

## Read
- This file, your issue, and the files it says you own. docs/statement.md
  if you touch what the proof binds. Nothing else unless the issue names it.

## Write
- Never append to a document. Your record is your pull request: what
  changed, the digest gate result on both backends, and the binding table
  if it moved.
- Standard words in prose, identifiers, flags and variables. No new nouns.
  GLOSSARY.md lists the identifiers we chose to keep; if you need another,
  rename instead.
- Design decisions the owner makes go in docs/decisions/, one dated file,
  never edited after merge.
- Commit only when asked. Never commit weights, witnesses, results or logs.

## Proofs
- The default configuration is the sound one. An --unsound-skip flag in a
  command is a red flag in a pull request title.
- A change that should not alter the proof reproduces the recorded digest
  on both backends before merge. A change that alters the proof records the
  new digest in prover/README.md in the same pull request.
- Every run JSON carries a provenance block; do not write numbers elsewhere.

## Hosts
- This repository knows no hosts. Measurements are made from zk-results.

## Reporting to the user
- Decisions for the user are a numbered list, one per item: the question in
  one plain sentence, what is at stake, the options, a recommendation.
- Say the thing, then the pointer. Never a bare item number, section
  number, constants key or identifier in a sentence addressed to the user.
- Consequence first, mechanism second, identifiers last.
- "Off by default", "asserted by the prover and checked by nothing" and
  "not built" are said in those words.
