# The digest gate

Three scripts that run the digest gate on a GPU box: the synthetic proofs on the
SIMD backend, then the same proofs on the CUDA backend, with the kernels
generated and installed in between.

| file | where it runs | what it does |
|---|---|---|
| `verify-branch.sh` | your machine | ships a tree to the box and runs the gate there |
| `on-card.sh` | the GPU box | the gate itself |
| `setup.sh` | the GPU box | prepares a fresh box, once |

Any CUDA card runs this gate. It has been run on a 24 GB card (sm_89) and on an
80 GB card (sm_90). Only the 8B reference proof needs 80 GB; the synthetic gate
does not.

## Running the gate

From a checkout of this repository, with the branch you want gated pushed:

```sh
tools/gate/verify-branch.sh my-branch            # gates origin/my-branch
tools/gate/verify-branch.sh --worktree           # gates the HEAD of this checkout
```

`--worktree` archives the HEAD commit, not the working tree: commit first, or
what you are testing is not what gets gated.

The script blocks until the gate is done, about half an hour, and prints the
log. If someone else is gating on the same box it waits for them first, so it
can take longer. Its exit status is the on-card script's; the test results are
in the log it prints, not in the status.

Settings, all optional:

| variable | default | meaning |
|---|---|---|
| `GATE_HOST` | `4090` | the ssh destination of the box |
| `GATE_SSH_OPTS` | empty | extra options for `ssh` and `scp` |
| `GATE_KEEP_DAYS` | `7` | `verify/<name>` trees older than this are deleted first |
| `FAM` | the list in `on-card.sh` | which kernel groups to generate |
| `NOIDLE` | `0` | `1` starts even if a timed run is using the card |

`GATE_HOST` is a `Host` entry in your `~/.ssh/config`. If you reach the box
through a shared connection instead, name the socket:

```sh
GATE_HOST=dummy GATE_SSH_OPTS="-o ControlPath=$HOME/.ssh/cm-4090 -o ControlMaster=no" \
    tools/gate/verify-branch.sh my-branch
```

Each gate uses `~/verify/<branch>` on the box: `src` is the shipped tree,
`on-card.sh` is the copy of the script this gate runs, `t-simd` and `t-cuda` are
the two target directories, `gen` is the generated kernel store, `logs` holds the
full build and test output, and `log` is everything the run printed.

## What the gate checks

In order:

1. Builds the SIMD backend and runs `cargo test --release --locked`. This is the
   whole test suite, and the synthetic digests are part of it.
2. Reads the shapes out of `prover/tests/synthetic_digest.rs` and writes one
   command line per shape. On `refactor-gadgets` that is 17 lines: the 16 cases
   in the table plus the one a test builds for itself.
3. Runs `cudaemit` and `genemit` for every shape, then installs the results into
   the vendored fork with the two scripts in `prover/kernels/`.
4. Builds the CUDA backend.
5. Checks that the built library really carries this card's architecture (see
   the trap below).
6. Runs the test suite again with `--features cuda`, on the device.

It also prints the phase times, the cgroup memory peak, and the size of the two
target directories.

The gate passes when both `cargo test` runs report `0 failed`, no line says
`digest moved`, and the last line says `VERIFY_DONE`. Paste those lines into
the pull request.

## When to run it

Every pull request on `refactor-gadgets` runs this gate on the 24 GB card,
before it is marked ready.

The 8B reference proof is the other check, and it needs an 80 GB card (the
device pool reaches 73 GB). It is not per pull request. Run it four
times: after phase 1, after phase 3, after phase 4, and once more before
`refactor-gadgets` is fast-forwarded into `main`. The phases are the ones listed
in issue #211.

## Disk and memory

100 GB of disk is comfortable. The toolkit is about 7 GB, rustup about 2 GB, the
cargo registry under 1 GB, the checkout under 0.5 GB, and each gated branch
keeps two target directories plus its generated store. Budget up to 2.3 GB per
target directory; the run of 2026-09-17 on a 24 GB card measured 699 MB for the
SIMD target, 862 MB for the CUDA target and 9.7 MB for the generated store, so a
handful of branches fits easily. Old trees are pruned by `GATE_KEEP_DAYS`.

32 GB of host RAM runs this gate: the 2026-09-17 run peaked at 16.9 GiB of the
62 GiB the container allowed, with other gates sharing the box at the time. Keep
64 GB free if you raise the `nvcc` parallelism over the generated kernels (issue
#190), which is what makes the peak large.

No model dump is needed on this host. The synthetic proofs need none, and the 8B
reference runs elsewhere.

For reference, that run took 30m25 in total: SIMD build 323 s, SIMD test 973 s
(711 s of it one control test), emit 19 s, CUDA build 347 s, CUDA test 162 s.

## The CMake fallback trap

The vendored library's CMake detects the card and sets the architectures it
compiles for. When detection fails it does not stop: it falls back to
`89;100;120`, and on any other card that produces a library which runs nothing.
The CUDA test then either fails oddly or silently exercises nothing of the
device path.

So the gate does not trust it. `on-card.sh` reads the card's compute capability
from `nvidia-smi` and exports it as `CUDAARCHS`, and after the build it runs
`cuobjdump --list-elf` on `libstwo_cuda.a` and stops with `FATAL` unless that
card's `sm_` code is in there. If you see that message, the detection failed;
check that `nvcc` and the driver are the versions `setup.sh` installed.

## Preparing a fresh box

On the box, as root, with this repository copied over or at least
`rust-toolchain.toml`:

```sh
bash tools/gate/setup.sh                       # reads ../../rust-toolchain.toml
bash tools/gate/setup.sh rust-toolchain.toml   # or name the file
bash tools/gate/setup.sh nightly-2025-07-14    # or the channel, if neither is there
```

It installs the build packages, the CUDA toolkit if `nvcc` is missing, rustup
and the pinned channel, and writes `TMPDIR` into `/etc/environment`. `TMPDIR`
matters: on these rentals `/tmp` sits on the container's overlay filesystem,
where `rustc` fails with `Bad address (os error 14)` writing its archives.
`GATE_TMPDIR` chooses the directory, `/root/tmp` by default; point it at a data
disk if `/root` is small.

The script prints `SETUP_DONE` when it is finished. Ubuntu 22.04 and 24.04 are
what it has been run on.
