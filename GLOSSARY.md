# Glossary

Identifiers kept although a standard word exists. Prose uses the standard word;
code and flags use the identifier. Each entry is one sentence; the list shrinks
when a rename is cheap.

- **chunk** — a segment: the computation of one layer (or of the output head)
  for `S` freshly decoded tokens against the committed key/value cache at
  context `C`, proved as one STARK; every recorded run has `S = 512`.
- **cell** — one committed field element of a trace column, the unit in which
  prover work is counted (columns times rows).
