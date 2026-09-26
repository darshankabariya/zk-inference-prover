#!/bin/bash
# setup.sh [<rust-toolchain.toml>|<channel>]  -- prepare a fresh Ubuntu 22.04 or
# 24.04 GPU box to run the digest gate. Run it once, as root, on the box itself.
#
# Installs the build packages, the CUDA toolkit if the box has none, rustup and
# the channel this repository pins, and puts TMPDIR on a data disk.
set -eux
export DEBIAN_FRONTEND=noninteractive

# /tmp on these rentals sits on the container's overlay root, where rustc dies with
# "Bad address (os error 14)" writing its archives. Keep temp on a data disk.
GATE_TMPDIR=${GATE_TMPDIR:-/root/tmp}
mkdir -p "$GATE_TMPDIR"
grep -q '^TMPDIR=' /etc/environment || echo "TMPDIR=\"$GATE_TMPDIR\"" >> /etc/environment
export TMPDIR=$GATE_TMPDIR

apt-get update
apt-get install -y --no-install-recommends \
    build-essential cmake curl ca-certificates git pkg-config python3 gnupg tmux rsync htop

# The toolkit is preinstalled on most rentals; install it only when nvcc is absent.
# cuobjdump comes with it, and the gate reads the built library with it.
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH
if ! command -v nvcc > /dev/null; then
    CUDA_PKG=${CUDA_PKG:-cuda-toolkit-12-9}
    . /etc/os-release
    DISTRO=ubuntu$(echo "$VERSION_ID" | tr -d .)   # 24.04 -> ubuntu2404
    curl -fsSL -o "$TMPDIR/cuda-keyring.deb" \
        "https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/x86_64/cuda-keyring_1.1-1_all.deb"
    dpkg -i "$TMPDIR/cuda-keyring.deb"
    apt-get update
    apt-get install -y --no-install-recommends "$CUDA_PKG"
    V=$(echo "${CUDA_PKG#cuda-toolkit-}" | tr - .)   # cuda-toolkit-12-9 -> 12.9
    if [ -d "/usr/local/cuda-$V" ]; then
        ln -sfn "/usr/local/cuda-$V" /usr/local/cuda
    else
        ln -sfn "$(ls -d /usr/local/cuda-* | sort -V | tail -1)" /usr/local/cuda
    fi
fi

# The pinned channel, read from rust-toolchain.toml rather than written out here,
# so this script cannot drift from the tree it gates. The argument is that file or
# a channel name; with no argument the file is taken from this checkout.
TC=${1:-$(dirname "$0")/../../rust-toolchain.toml}
if [ -f "$TC" ]; then
    CHANNEL=$(sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$TC" | head -1)
else
    CHANNEL=$TC   # a channel name given directly, for a box without the checkout
fi
[ -n "$CHANNEL" ] || { echo "no channel: pass rust-toolchain.toml or a channel name" >&2; exit 1; }

# rustup-init refuses to run over an existing installation, so this stays out of
# the way on a re-run.
command -v rustup > /dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
# the channel is installed explicitly: the gate runs with RUSTUP_AUTO_INSTALL=0,
# which blocks rustup from fetching a missing channel on first use.
rustup toolchain install "$CHANNEL" --profile minimal
rustup default "$CHANNEL"

grep -q 'CUDAHOSTCXX' ~/.bashrc || cat >> ~/.bashrc <<RC
export PATH=/usr/local/cuda/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH
export CUDAHOSTCXX=/usr/bin/g++
export TMPDIR=$GATE_TMPDIR
RC

cmake --version | head -1
rustc --version
cargo --version
nvcc --version | tail -2
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
echo SETUP_DONE
