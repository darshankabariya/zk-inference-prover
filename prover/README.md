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

Every digest below was taken after the binding work of 2026-09-03 to 2026-09-06:
the matmul-output bindings, the norm scale, the SwiGLU product and the rotation
are unconditional, and the preprocessed trace is two committed trees (a shape
tree whose root the verifier derives from the public parameters, and a model
tree whose root is compared with the published model commitment). The layer
boundary, the K/V boundary, the argmax and the embedding gadget were still off
by default when these were taken; the rows say which of them a command turns
on. When the sound defaults land this table gains the sound digests and keeps
these under "unbound configuration", reproduced with every skip flag set.

The synthetic shapes share one base command and differ only in the flags after
it. `GKR5` stands for the five flags
`--gkr-bands --gkr-dequant --gkr-glue --gkr-walks --gkr-attnout`.

```sh
SYN="--round away --variants intsplit6 --table-bits 16 --bands 2,1 --s 32 --ctx 128 --hd 32"
GLUE="--nout 64 --glue-a 32 --kmax 16 --synth-glue"
b65 compose $SYN <flags>
```

| shape | flags after `$SYN` | digest (first 16 hex) | recorded |
|---|---|---|---|
| `corner` | `--corner` | `164079c6a045df22` | SIMD backend, 2026-09-06; pinned in `tests/` |
| `flagoff` | (none) | `156285f2f70eaa80` | same |
| `kvbind` | `--corner --kv-bind` | `4d3835c942072f67` | same |
| `gkrbands` | `--corner --gkr-bands` | `c44ce8a832a9bec7` | same |
| `gkrfive` | `--corner GKR5` | `5d7aa189c702c2a5` | same |
| `kv_gkrbands` | `--corner --kv-bind --gkr-bands` | `b857c0b7a978642f` | same |
| `kv_gkrfive` | `--corner --kv-bind GKR5` | `a379fa16dc2b0e98` | same |
| `glue_corner` | `$GLUE --corner` | `5a4a05ea6274e7c6` | same |
| `glue_kv` | `$GLUE --corner --kv-bind` | `94c32161ecbbef84` | same |
| `glue_bounds` | `$GLUE --corner --bounds` | `775b1972badf31e4` | same |
| `glue_gkrfive` | `$GLUE --corner GKR5` | `d1aff3e6580569a4` | same |

| proof | command | digest (first 16 hex) | recorded |
|---|---|---|---|
| Llama-3.1-8B layer 16, `S = 512`, `C = 16384` | `RAYON_NUM_THREADS=16 b65 compose --round away --numbits 9 --z-hi-bits 7 --variants intexp --table-bits 25 --bands 16,8,8 --witness <dir>` | `42fceb1f3a8643cd` | both backends, 2026-09-06 |

The 8B witness is one layer's dump written by `harness.witness.extract` under
the `b10-final` quantization configuration; `zk-results` records which run
produced it. The synthetic cases need nothing but the binary and take seconds
each. Older values of the 8B digest (`6c3fb61524dee2ec` before the binding
work, `616d6a80ca9c003f` before the tree split) are history and reproduce on
no current commit.

## Configuration in the JSON

Every run prints one JSON document: `env` (threads, memory, GPU as the machine
reports it), `modes` (every device switch with its value and whether the
environment set it), `security` (the FRI parameters and the hash),
`provenance`, and `result`. A recorded number must be read together with these
blocks; the environment string alone does not name a configuration.

`provenance` says what produced the document: the source `commit` and whether
the tree was `dirty` (both from `build.rs`; `unknown` when the build had no
git and no `.source-commit` file), the `binary` name and its `args`, the
`backend`, the `gpu`, the `driver_version` and the `cuda_version` the driver
reports, the `proof_sha256` (one proof) or `proof_sha256s` (a chunk loop),
the `unsound_skips` in effect, and the `binding` table folded over every proof
the run made. None of it enters a statement or a proof, so the digest does not
depend on it.
