# zk-prover

A STARK prover for the decode step of a quantized transformer, and the Python
harness that produces its witnesses and quality numbers. Software only: no
measured result and no knowledge of any host lives here. Measurements are made
from the companion repository `zk-results`, which pins the commit of this one
that produced every run.

## Layout

| path | what |
|---|---|
| `prover/` | the prover crate, binary `zk-prover`; `prover/README.md` lists the recorded proof digests; `prover/kernels/` installs generated device kernels into the CUDA backend |
| `nitroo/` | the vendored Circle STARK library the prover builds on; its `README.md` names the upstream commit and each local change |
| `harness/` | one uv package: `witness/` (deterministic quantized forward pass, greedy continuation, witness extraction, shape table, plan file), `eval/` (perplexity, task suite, the unproved baseline), `costmodel/` |
| `docs/statement.md` | what the proof binds, edge by edge, and which `--unsound-skip-<name>` flag unbinds each |
| `docs/decisions/` | prover design decisions, one dated file each, never edited after merge |
| `GLOSSARY.md` | the identifiers kept despite not being standard |
| `Dockerfile` | both backends from a clean image |

## Building

The prover has two backends selected by one cargo feature. Without the feature
the same code runs on the SIMD backend, which is the reference every digest is
compared against; with `--features cuda` commitment and proving run on the
device, at device speed once `prover/kernels/` has installed the generated
kernels for the shapes being proved. The vendored library links its CUDA library on every Linux build, so
the CUDA 12.9 toolkit is needed at build time even for the SIMD backend. No GPU
is needed to build either one.

```sh
cargo build --release                                   # SIMD backend
cargo build --release -p zk-prover --features cuda   # CUDA backend
```

Requirements: the toolchain pinned in `rust-toolchain.toml` (rustup selects
it), `cmake`, and the CUDA 12.9 toolkit with a host compiler nvcc accepts (gcc
14 or older). Set `CUDAARCHS=90` for the recorded runs' device architecture.
The `Dockerfile` does all of this:

```sh
docker build -t zk-prover .
```

On macOS the crate type-checks (`cargo check --release`, with or without the
feature) but does not link, because the CUDA library is not available there.

## Proving one synthetic segment

Without `--witness` the prover builds a small synthetic layer segment and
proves it. This is the shape check CI pins:

```sh
./target/release/zk-prover compose --s 32 --ctx 64 --bands 2,1 --corner
```

The JSON on stdout carries `result.point.accepts` and `result.point.proof_sha256`.
The expected digest is in `prover/README.md` and in `prover/tests/`.

## The digest gate

A change that should not alter the proof reproduces the recorded digests on
both backends before it merges. A change that alters the proof records the new
digests in `prover/README.md` in the same pull request. `cargo test` runs the
synthetic cases; the full-size reference needs a witness and a GPU host, which
`zk-results` provides.

## Soundness switches

Every binding gadget is on by default wherever the witness lets it be stated,
and a witness that cannot state one is refused. `--unsound-skip-<name>` turns
one off, loudly: the name is listed in the JSON's `provenance.unsound_skips`
and the binding table (`provenance.binding`, and `binding` on every proof)
marks the edge unbound. The names are `residual-stream`, `layer-boundary`,
`kv-boundary`, `argmax` and `embedding`; `docs/statement.md` says what each
one gives up. A run with a skip in effect is a control, never a result.

## The harness

```sh
cd harness && uv sync
uv run python -m harness.witness.selftest
```

Every script runs as a module. Scripts that read measured numbers take
`--constants PATH`, a `constants.json` from `zk-results`, and `--artifacts DIR`
or `--inputs DIR` for run outputs. See `harness/README.md`.

## Rules for agents

`CLAUDE.md`. In one line: read your issue and the files it names, write standard
words, never append to a document, and treat an `--unsound-skip` flag in a
command as a red flag.
