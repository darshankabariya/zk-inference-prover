# Lookup tables keyed by a bf16 value keep bf16 keys

Date: 2026-09-03. Status: decided by the owner.

## Decision

No lookup table the AIR indexes by a bf16 value moves to an integer key. This
covers the softmax's exponential table and the normalisation's reciprocal
table, and any table added later whose input is a bf16 intermediate of the
quantized forward pass.

## Why

The harness scores the model on exactly the arithmetic the proof constrains:
the same bf16 intermediates, the same roundings, in the same order. Re-keying
a table by an integer changes that arithmetic, so the quality numbers would no
longer describe the proved computation, and every perplexity and task result
would have to be measured again on the new arithmetic before it could be
cited. A cheaper table is not worth a second quantization configuration.

## Consequence

A change that wants a smaller or faster table changes how the bf16 key is
formed or how the table is committed, not what it is keyed by.
