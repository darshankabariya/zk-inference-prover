# prover

One crate, two backends. Without the `cuda` feature the code runs on the SIMD
backend, the reference every digest is compared against; with it, commitment
and proving move onto the vendored library's device backend. The two builds
share every line of witness and trace construction, the hash channel, the
proof serialization and the digest.

```sh
cargo build --release                                   # SIMD backend
cargo build --release -p zk-prover --features cuda   # CUDA backend
```

The vendored library in `../nitroo` links its CUDA library on every Linux
build, so the CUDA toolkit is needed to build either backend; a GPU is needed
only to run the CUDA one. On macOS the crate type-checks but does not link.
A fresh CUDA build evaluates constraints on the CPU until `kernels/` has
installed the generated kernels for the shapes being proved; digests do not
depend on it.

## Repeated flags

Every subcommand reads its command line left to right and **the last occurrence
of a flag wins**: under `--table-bits 25 --table-bits 16` the width is 16,
everywhere. So a shared prefix is written first and what one segment decides for
itself after it, and `compose-chunks` composes a plan line the same way — the
command line, then the line — which is what makes `compose $SHARED $line`
reproduce the digest the loop gave that line. A flag that carries no value, such
as `--diagonal-block`, is read as present or absent, so repeating one says
nothing new.

## The digest gate

Any change that is not meant to alter the proof reproduces every digest below
on both backends before it merges. A change that alters the proof records the
new digests here, in the same pull request, and says why. The digest is
`result.point.proof_sha256` in the JSON the binary prints on stdout.

## Soundness switches

Every binding gadget is on by default wherever the witness lets it be stated:
the residual-stream components (the norms, the residual adds, the requantize
seams, the dequant output bindings, the rotation), the boundary trees over the
residual stream and over the keys and values, the argmax on a head segment and
the embedding gadget on a layer-0 segment. A witness that cannot state one is
refused with a message naming the skip that gets past the refusal:

| `--unsound-skip-<name>` | what it gives up | when a witness forces it |
|---|---|---|
| `residual-stream` | the residual-stream components: edges 1 to 3 and 5 to 11 of the binding table | a dump extracted without `--residual-stream`, or a bare synthetic segment |
| `layer-boundary` | the residual-stream boundary trees (edge 13) | never |
| `kv-boundary` | the K/V boundary trees (edge 14) | a segment without its own block (`--diagonal-block`) |
| `argmax` | the head's argmax (edge 16) | a head dump without `key_head.u16`, or a synthetic head |
| `embedding` | the layer-0 embedding gadget (edge 15) | a layer-0 dump extracted without `--embed` |

One switch weakens a gadget instead of turning it off. `--unsound-argmax-lax`
proves that the emitted token attains the maximum of the logits where the
default proves that it is the lowest-indexed maximum, which is the rule the
model decodes by; on a position whose two highest logits are equal -- 15 of 512
at a context of 4096 on the real continuations -- the weaker statement admits
either token. It is off by default, it is refused on a statement that states no
argmax, it lists `argmax-lax` in `provenance.unsound_skips`, and the binding
table marks edge 16 unbound with that reason while it is on, so a run that used
it is visibly a control. The name it had before, `--argmax-lax`, is an error.

The skips in effect are listed in `provenance.unsound_skips`; an unknown name
is an error. The flags these replaced (`--bounds`, `--kv-bind`, `--embed`,
`--head-argmax`, `--head-key`, `--norm-tails`, `--swiglu`, `--deq-ties`,
`--no-glue`) are retired and reported once on stderr when seen. The five
`--gkr-*` flags are not soundness switches: they choose the LogUp-GKR backend
for a component and change the proof, not the statement.

## Recorded digests

Every digest below was taken after the binding work of 2026-09-03 to 2026-09-06
and the sound defaults of 2026-09-07. The preprocessed trace is two committed
trees (a shape tree whose root the verifier derives from the public parameters,
and a model tree whose root is compared with the published model commitment).

The synthetic shapes share one base command and differ in the flags after it.
`RESIDUAL_STREAM` builds the residual-stream components on the synthetic
segment and `GKR5` stands for the five flags
`--gkr-key-tiles --gkr-dequant --gkr-residual-stream --gkr-freivalds --gkr-attnout`.

```sh
SYN="--round away --variants intsplit6 --table-bits 16 --s 32 --ctx 128 --hd 32"
RESIDUAL_STREAM="--nout 64 --residual-stream-a 32 --kmax 16 --synth-residual-stream"
zk-prover compose $SYN <flags>
```

**The sound defaults.** Three heads of 32 over 64-wide matmuls cannot state the
rotation or the o seam (edges 3 and 5 stay unbound on that shape, as the
binding table says); four heads over 128-wide matmuls bind every layer edge, and
so do five heads over 160-wide ones, which is the shape whose head count tiles
no power of two.

| shape | flags after `$SYN` | digest (first 16 hex) | unbound edges | recorded |
|---|---|---|---|---|
| `sound` | `--head-groups 2,1 $RESIDUAL_STREAM --diagonal-block` | `82085896a8408f36` | 3, 5 | both backends, 2026-09-07; pinned in `tests/` |
| `sound_gkrfive` | `--head-groups 2,1 $RESIDUAL_STREAM --diagonal-block GKR5` | `6bee5972d6f616aa` | 3, 5 | same |
| `sound_aligned` | `--head-groups 2,2 --nin 128 --nout 128 --residual-stream-a 32 --kmax 16 --synth-residual-stream --diagonal-block` | `9c3cdebd97a7d57b` | none | same |
| `sound_padded` | `--head-groups 4,1 --nin 160 --nout 160 --residual-stream-a 32 --kmax 16 --synth-residual-stream --diagonal-block` | `284e5f3d24ad2c52` | none | SIMD backend, 2026-09-08; pinned in `tests/`; **new**, not moved |

**The padded head count.** `sound_padded` is the last row's shape and is new
here, not a moved digest: five heads of 32 over 160-wide matmuls. Thirty-two
tokens times five heads is 160, and no block length makes that a power of two,
so before the decision of 2026-09-08 the rotation refused the layer outright
(issue #73, and Qwen2.5-14B's 40 query heads are the same arithmetic). The
rotation now holds 256 rows for its 160 real ones and each quantize component
512 for its 320, with the difference carried at the front as unconstrained
padding: rows that hold zero, that no constraint says anything about, and that
the two Horner comparisons binding the real rows check are zero. Every layer
edge is bound on this shape and the binding table's edge 3 says what the padding
is; the padded rows are 37.5% of the rotation's and the quantize's own committed
cells, which is 230,400 cells of the segment's 7,066,144. The three rows above
tile exactly, take the same path they always did, and their digests do not move.

Reading the first three on the CUDA backend took one fix in the kernel emitter. A
generated LogUp kernel used to ask for exactly as many preprocessed columns as
the evaluator it was emitted from carries, and the evaluator the prover lowers
for these lookups leaves off two accumulator sides whose selectors no lookup
reads, so the two counts differ and the proof stopped before the first kernel
ran. The kernel now asks that every column it reads was supplied, which is what
its fingerprint pins; no digest moved.

**The head shape.** A synthetic head segment: the final norm, an `lm_head`
matmul over its output, and the layer's own seven matmuls beside them, which is
what a head dump carries. Its one requantize seam is the head's, so the seam
between the attention output and the `o` matmul is not part of what it states
and edge 5 is **absent** there rather than unbound -- as are edges 3, 4, 6 to 11
and 14. A synthetic head carries no `key_head.u16`, so the argmax cannot be
stated on it and `--unsound-skip-argmax` is what a synthetic head forces; edge 16
is unbound here for that reason alone, and on a head dump, which does carry it,
that edge is bound and nothing else in the table differs. The row is a shape no
page pinned before, so it is a new digest and not a moved one.

| shape | flags after `$SYN` | digest (first 16 hex) | unbound edges | recorded |
|---|---|---|---|---|
| `head` | `--head-groups 2,1 --nout 64 --residual-stream-a 32 --kmax 16 --synth-head --synth-vocab 256 --unsound-skip-argmax` | `e3f31c1d4aeae77e` | 16 | SIMD backend, 2026-09-07; pinned in `tests/` |

**The unbound configuration.** The same shapes with the skips set reproduce the
digests recorded on 2026-09-06, before the sound defaults; `RS`, `LB` and `KV`
stand for `--unsound-skip-residual-stream`, `--unsound-skip-layer-boundary` and
`--unsound-skip-kv-boundary`.

| shape | flags after `$SYN` | digest (first 16 hex) | recorded |
|---|---|---|---|
| `diagonal_block` | `--head-groups 2,1 --diagonal-block RS KV` | `164079c6a045df22` | SIMD backend, 2026-09-06; pinned in `tests/` |
| `flagoff` | `--head-groups 2,1 RS KV` | `156285f2f70eaa80` | same |
| `kvbind` | `--head-groups 2,1 --diagonal-block RS` | `4d3835c942072f67` | same |
| `gkr_key_tiles` | `--head-groups 2,1 --diagonal-block RS KV --gkr-key-tiles` | `c44ce8a832a9bec7` | same |
| `gkrfive` | `--head-groups 2,1 --diagonal-block RS KV GKR5` | `5d7aa189c702c2a5` | same |
| `kv_gkr_key_tiles` | `--head-groups 2,1 --diagonal-block RS --gkr-key-tiles` | `b857c0b7a978642f` | same |
| `kv_gkrfive` | `--head-groups 2,1 --diagonal-block RS GKR5` | `a379fa16dc2b0e98` | same |
| `residual_stream_diagonal_block` | `--head-groups 2,1 $RESIDUAL_STREAM --diagonal-block LB KV` | `5a4a05ea6274e7c6` | same |
| `residual_stream_kv` | `--head-groups 2,1 $RESIDUAL_STREAM --diagonal-block LB` | `94c32161ecbbef84` | same |
| `residual_stream_bounds` | `--head-groups 2,1 $RESIDUAL_STREAM --diagonal-block KV` | `775b1972badf31e4` | same |
| `residual_stream_gkrfive` | `--head-groups 2,1 $RESIDUAL_STREAM --diagonal-block LB KV GKR5` | `d1aff3e6580569a4` | same |

**The full-size reference.** One Llama-3.1-8B layer-16 segment, `S = 512`
output positions against a `C = 16384` key prefix, from a dump
`harness.witness.extract` writes under the fixed quantization configuration
`b10-final`:

```sh
uv run --no-group cpu --group cuda python -m harness.witness.extract \
    --tokens 16896 --chunk 512 --layer 16 --spec b10-final \
    --residual-stream --diagonal-block --out <dir>
RAYON_NUM_THREADS=16 zk-prover compose --round even --numbits 9 --z-hi-bits 7 \
    --variants intexp --table-bits 25 --head-groups 16,8,8 --diagonal-block $GKR5 \
    --witness <dir>
```

| proof | digest (first 16 hex) | recorded |
|---|---|---|
| Llama-3.1-8B layer 16, `S = 512`, `C = 16384` | `25741d12f315aae7` | both backends, 2026-09-07; no skip in effect, every edge bound, no kernel fallback; `zk-results` run `2026-09-07-reference-digest-h100` |

The dump carries its own diagonal block and the residual-stream columns, so
every binding gadget the defaults switch on is stated on it and the proof
needs no skip flag. Two flags on the prover side are not free choices.
`--round even` is the dump's own mode: `b10-final` rounds a half to even and
the dump's `dequant_round` says so, the loader compares the two and refuses a
command naming the other mode, which would build tables whose keys this
witness's values miss and be rejected after the whole prove. The five GKR
reductions are on because without them this segment's interaction tree is
93,184 columns wide, and the vendored library's out-of-domain sampling puts
the polynomial count in a thread block grid's y dimension, which a device caps
at 65,535: the CUDA backend stops in `eval_at_point.cu` with an invalid
configuration argument. With them the interaction tree is 5,600 columns.

**The reference's unbound predecessor.** The dump below was extracted without
its own diagonal block and predates the embedding rows, so under the defaults
it proves only with the K/V boundary skipped and its digest is an unbound one.
No host holds that dump now, so it reproduces nowhere; the row is kept as the
record of what that configuration gave.

| proof | command after `zk-prover compose` | digest (first 16 hex) | recorded |
|---|---|---|---|
| Llama-3.1-8B layer 16, `S = 512`, `C = 16384` | `--round away --numbits 9 --z-hi-bits 7 --variants intexp --table-bits 25 --head-groups 16,8,8 --unsound-skip-kv-boundary --witness <dir>` | `42fceb1f3a8643cd` | both backends, 2026-09-06, before the sound defaults |

That command names `--round away` while `b10-final` rounds a half to even, so
either the dump was written under an earlier quantization configuration or the
row's command and its dump never matched. Nothing on any host can settle it,
and no page here should read the row as an extraction recipe: the loader now
refuses that pairing.

The synthetic cases need nothing but the binary and take seconds each. Older
values of the 8B digest (`6c3fb61524dee2ec` before the binding work,
`616d6a80ca9c003f` before the tree split) are history and reproduce on no
current commit.

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
