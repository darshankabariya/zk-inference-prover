#!/bin/bash
# verify-branch.sh <branch>             -- gate origin/<branch>
# verify-branch.sh --worktree [<name>]  -- gate the HEAD commit of this checkout
#
# Ships the tree to the gate host as a tar archive, runs tools/gate/on-card.sh
# there under a lock, and prints its log. Blocks until the gate is done, which is
# about half an hour on a 24 GB card, longer if another gate holds the lock.
# Exits with the on-card script's status; read the log for the test results.
#
#   GATE_HOST       ssh destination (default: 4090)
#   GATE_SSH_OPTS   extra options for ssh and scp, e.g. a shared connection:
#                   -o ControlPath=~/.ssh/cm-4090 -o ControlMaster=no
#   GATE_KEEP_DAYS  remove verify/<name> trees older than this many days (default: 7)
#   FAM, NOIDLE     passed through to on-card.sh
set -u
SELF=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SELF/../.." && pwd)
cd "$REPO" || exit 1
H=${GATE_HOST:-4090}
SSH_OPTS=${GATE_SSH_OPTS:-}
DAYS=${GATE_KEEP_DAYS:-7}

case ${1:-} in
  --worktree)
    REF=HEAD
    B=${2:-$(git symbolic-ref --quiet --short HEAD)}
    [ -n "$B" ] || { echo "detached HEAD: give a name, or gate a branch" >&2; exit 1; } ;;
  ""|-h|--help)
    sed -n '2,3p' "$0" >&2; exit 1 ;;
  *)
    B=$1; REF=origin/$B
    git fetch -q origin || exit 1 ;;
esac
NAME=$(echo "$B" | tr '/' '-')
# $NAME is interpolated into commands the gate host runs, including an rm -rf, so
# it has to be a plain path element. tr has already folded the slashes; reject
# anything else that is not a letter, digit, dot, dash or underscore.
case $NAME in
  ""|.|..|*[!A-Za-z0-9._-]*)
    echo "refusing the name \"$NAME\" (from branch \"$B\"): it must be letters, digits, '.', '-' or '_'" >&2
    exit 1 ;;
esac
COMMIT=$(git rev-parse --verify "$REF") || exit 1
echo "== gating $REF ($COMMIT) as $NAME on $H"

# scp resolves a relative remote path against the directory the box's sftp server
# starts in, which is not always the home directory, so ask the box for its home
# and write full paths. Commands run over ssh do start in the home directory.
# shellcheck disable=SC2086  # GATE_SSH_OPTS is a list of options
RH=$(ssh $SSH_OPTS "$H" 'echo $HOME' | tr -d '\r')
[ -n "$RH" ] || { echo "no home directory on $H" >&2; exit 1; }

# Each gate leaves two target directories of about 2.3 GB plus the generated
# store, so drop the trees nobody has looked at for a while before shipping.
# shellcheck disable=SC2086
ssh $SSH_OPTS "$H" "mkdir -p $RH/verify && find $RH/verify -mindepth 1 -maxdepth 1 -type d -mtime +$DAYS -print -exec rm -rf {} +" | sed 's/^/== pruned /'

# shellcheck disable=SC2086
git archive --format=tar "$REF" | ssh $SSH_OPTS "$H" "rm -rf $RH/verify/$NAME/src && mkdir -p $RH/verify/$NAME/src && tar -C $RH/verify/$NAME/src -xmf - && echo '$COMMIT ($REF)' > $RH/verify/$NAME/src/.source-commit" || exit 1
# The on-card script goes next to the tree it gates, not to one shared path: bash
# reads a script as it runs it, so overwriting the copy another gate is running
# would break that gate.
# shellcheck disable=SC2086
scp -q $SSH_OPTS "$SELF/on-card.sh" "$H:$RH/verify/$NAME/on-card.sh" || exit 1

# flock serialises against any other gate on the same box; the card is not shared.
# shellcheck disable=SC2086
ssh $SSH_OPTS "$H" "FAM='${FAM:-}' NOIDLE=${NOIDLE:-0} flock $RH/verify.lock bash $RH/verify/$NAME/on-card.sh $NAME > $RH/verify/$NAME/log 2>&1; rc=\$?; cat $RH/verify/$NAME/log; exit \$rc"
