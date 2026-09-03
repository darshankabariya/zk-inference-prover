# prover

One crate, two backends. Without the `cuda` feature the code runs on the SIMD
backend, the reference every digest is compared against; with it, commitment
and proving move onto the vendored library's device backend. The two builds
share every line of witness and trace construction, the hash channel, the
proof serialization and the digest.

```sh
cargo build --release                                   # SIMD backend
cargo build --release -p b65-pipeline --features cuda   # CUDA backend
```

The vendored library in `../nitroo` links its CUDA library on every Linux
build, so the CUDA toolkit is needed to build either backend; a GPU is needed
only to run the CUDA one. On macOS the crate type-checks but does not link.
A fresh CUDA build evaluates constraints on the CPU until `kernels/` has
installed the generated kernels for the shapes being proved; digests do not
depend on it.

## The digest gate

Any change that is not meant to alter the proof reproduces every digest below
on both backends before it merges. A change that alters the proof records the
new digests here, in the same pull request, and says why. The digest is
`result.point.proof_sha256` in the JSON the binary prints on stdout.

## Recorded digests

All three were recorded with every binding gadget, the layer boundary and the
argmax **off**, which was the state of every measured run to date. When the
sound defaults land, this table gains the sound digests and keeps these under
"unbound configuration", reproduced with every skip flag set.

| proof | command | digest (first 16 hex) | recorded |
|---|---|---|---|
| synthetic segment, `S = 32`, `C = 64` | `b65 compose --s 32 --ctx 64 --bands 2,1 --corner` | `34e30fa70297e2a9` | both backends; pinned in `tests/` |
| synthetic segment, `S = 128`, `C = 256` | `b65 compose --s 128 --ctx 256 --bands 2,1 --corner` | `37421c358a7a0f21` | both backends; pinned in `tests/` |
| Llama-3.1-8B layer 16, `S = 512`, `C = 16384` | `RAYON_NUM_THREADS=16 b65 compose --round away --numbits 9 --z-hi-bits 7 --variants intexp --table-bits 25 --bands 16,8,8 --witness <dir>` | `6c3fb61524dee2ec` | both backends, 2026-08-24 |

The 8B witness is one layer's dump written by `harness.witness.extract` under
the `b10-final` quantization configuration; `zk-results` records which run
produced it. The synthetic cases need nothing but the binary and take seconds.

## Configuration in the JSON

Every run prints one JSON document: `env` (threads, memory, GPU as the machine
reports it), `modes` (every device switch with its value and whether the
environment set it), `security` (the FRI parameters and the hash), and
`result`. A recorded number must be read together with these blocks; the
environment string alone does not name a configuration. A `provenance` block
(commit, command line, driver, digest, skips in effect, binding table) is the
next addition.
