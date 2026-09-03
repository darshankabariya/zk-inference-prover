# harness

Witness generation (`witness/`), quality evaluation (`eval/`) and the cost
model (`costmodel/`), as one uv package. Run every script as a module from this
directory, after `uv sync`:

    uv run python -m harness.witness.selftest

Scripts that read measured numbers take `--constants PATH` (a zk-results
`constants.json`) and `--artifacts DIR` or `--inputs DIR` for run outputs.
Nothing measured lives in this repository. `witness/text.txt` is the fixed
evaluation corpus (Pride and Prejudice, Project Gutenberg #1342).

## The vLLM baseline's environment

`eval/baseline` measures the unproved decode baseline in two backends. The `hf`
backend runs in this environment. vLLM cannot share it (it pins its own CUDA
torch), so it gets a separate virtual environment that is never committed:
`uv venv .venv-vllm && uv pip install --python .venv-vllm vllm`, then run
`decode_bench --backend vllm` with `.venv-vllm/bin/python -m harness.eval.baseline.decode_bench`.
