"""The provenance block a harness output carries.

A prover output's provenance says which commit made the proof, on which
backend, and what the proof binds; `prover/src/util.rs` writes it into every
run JSON, and `CLAUDE.md`'s rule is that every run JSON carries one.

A harness output is a run JSON too, and it makes no proof.  It measures the
arithmetic the prover constrains -- a perplexity, a task score, a cosine
similarity -- on a named model at a named revision, under a named quantization
variant, over a named corpus or dataset.  So it carries a block of the same
shape and for the same purpose, with the fields that mean something here filled
in, and the two that cannot mean anything here stated as empty rather than left
out: `binding` is null and there is no proof digest.  `kind` is `harness`, so
that a reader -- and any record that reads these files -- can tell the two kinds
of run apart rather than having to infer it from a missing field.

    from .. import provenance
    doc = {"provenance": provenance.block(model=MODEL, device=DEVICE,
                                          threads=THREADS, module=__spec__.name,
                                          spec=name, corpus=provenance.digest(TEXT)),
           "result": rows}

`commit` is read the way the prover's build script reads it: from
`.source-commit`, the file written beside a tree that was pushed to a host
without its `.git`, and from `git` when this is a checkout instead.  Both may be
null, and a null commit is a fact about the run, not an error.
"""
import hashlib
import os
import pathlib
import platform
import subprocess
import sys


def _from_source_commit():
    """`.source-commit`: the commit a pushed tree came from, and whether it was dirty.

    Written beside the sources, so it is at a parent of this file.  The search
    starts at this file rather than at the working directory, because the
    working directory of a harness process is the project and not the tree.
    """
    for start in (pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()):
        for d in (start, *start.parents):
            f = d / ".source-commit"
            try:
                if f.is_file():
                    parts = f.read_text().split()
                    if parts:
                        return parts[0], "dirty" in parts[1:]
            except OSError:
                pass
    return None, None


def _from_git():
    here = str(pathlib.Path(__file__).resolve().parent)
    try:
        r = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None, None
        s = subprocess.run(["git", "-C", here, "status", "--porcelain",
                            "--untracked-files=no"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip(), bool(s.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def source():
    """(commit, dirty), either of which may be null."""
    commit, dirty = _from_source_commit()
    if commit is None:
        commit, dirty = _from_git()
    return commit, dirty


def _smi(query):
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _cuda_version():
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
        for line in r.stdout.splitlines():
            if "CUDA Version:" in line:
                return line.split("CUDA Version:")[1].split()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def digest(path):
    """A file's name, sha256 and length: what names a corpus or a dataset exactly.

    A quality number is only reproducible against the bytes it was measured
    over, and a dataset that is downloaded rather than carried has no other
    version to name.
    """
    name = os.path.basename(path)
    try:
        h, n = hashlib.sha256(), 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
                n += len(chunk)
        return {"file": name, "sha256": h.hexdigest(), "bytes": n}
    except OSError:
        return {"file": name, "sha256": None, "bytes": None}


def model_revision(model):
    """The commit of the model repository the weights were read from, if it is known."""
    try:
        from huggingface_hub import constants as hub
        ref = (pathlib.Path(hub.HF_HUB_CACHE) / ("models--" + model.replace("/", "--"))
               / "refs" / "main")
        if ref.is_file():
            return ref.read_text().strip()
    except Exception:
        pass
    return None


def _versions():
    v = {"python": platform.python_version()}
    for name in ("torch", "transformers", "numpy"):
        try:
            v[name] = __import__(name).__version__
        except Exception:
            v[name] = None
    return v


def rows(doc):
    """The `result` of a harness output, whichever shape the file is in.

    A file written before this block existed is the bare result; one written
    since carries `{provenance, result}` like every other run JSON.  Readers go
    through here so that an output from an earlier session still reads.
    """
    if isinstance(doc, dict) and "provenance" in doc and "result" in doc:
        return doc["result"]
    return doc


def block(*, model, device, threads, module, argv=None, **extra):
    """The block, with whatever the caller measured named in `extra`.

    `extra` is where a caller puts what its own numbers are about: the
    quantization variant (`spec`), the corpus or the datasets it read, the
    number of chunks. Nothing here reads them; they are recorded.
    """
    commit, dirty = source()
    return {
        "kind": "harness",
        "commit": commit,
        "dirty": dirty,
        "binary": f"python -m {module}",
        "args": list(sys.argv[1:] if argv is None else argv),
        "backend": "harness",
        "gpu": _smi("name,memory.total"),
        "driver_version": _smi("driver_version"),
        "cuda_version": _cuda_version(),
        "device": device,
        "threads": threads,
        "versions": _versions(),
        "model": model,
        "model_revision": model_revision(model),
        # A harness run proves nothing, and says so rather than being silent
        # about it: no proof was made, so there is no digest and no edge of the
        # statement is bound by anything here.
        "proof_sha256": None,
        "proof_sha256s": None,
        "unsound_skips": [],
        "binding": None,
        **extra,
    }
