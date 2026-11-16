import ast
import builtins
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "run_kaggle.ipynb"
# Setup cells, eval-phase setup, then the judge phase: the documented judge-only path.
JUDGE_ONLY_PATH = [
    "cell-1",
    "cell-2",
    "acae54e37e7d407bbb7b55eff062a284",
    "cell-9",
    "cell-10",
    "cell-11",
    "t17c-tuning-run",
    "cell-12",
]


def _tree(cell: dict) -> ast.Module:
    lines = "".join(cell["source"]).splitlines()
    return ast.parse("\n".join(line for line in lines if not line.startswith("!")))


def test_judge_only_path_defines_every_name_it_uses():
    cells = {c["id"]: c for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]}
    defined = set(dir(builtins))
    late: set[str] = set()
    for cell_id in JUDGE_ONLY_PATH:
        tree = _tree(cells[cell_id])
        # Coarse: names bound anywhere in a cell (scopes ignored) count as defined afterwards.
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.FunctionDef | ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
            elif isinstance(node, ast.Import | ast.ImportFrom):
                defined.update((a.asname or a.name).split(".")[0] for a in node.names)
        # Names read inside a def resolve at call time, so they only need a definition by the end.
        in_def = {
            id(n) for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) for n in ast.walk(f)
        }
        loads = [
            n for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        ]
        late |= {n.id for n in loads if id(n) in in_def}
        used = {n.id for n in loads if id(n) not in in_def}
        assert used <= defined, f"{cell_id} uses undefined {sorted(used - defined)}"
    assert late <= defined, f"functions use undefined {sorted(late - defined)}"


def test_judge_server_uses_both_gpus_with_tensor_parallelism():
    cells = {c["id"]: c for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]}
    src = "".join(cells["cell-10"]["source"])
    assert '"--tensor-parallel-size"' in src and '"CUDA_VISIBLE_DEVICES": "0,1"' in src


def test_eval_runs_derive_from_split_and_use_final_config():
    cells = {c["id"]: c for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]}
    src = "".join(cells["acae54e37e7d407bbb7b55eff062a284"]["source"])
    assert 'SPLIT = "test"' in src and "FINAL_CONFIG" in src
    for name in ("e1_dense", "e1_rerank", "e2_none", "e2_prompt", "e2_full"):
        assert f'f"{name}_{{SPLIT}}"' in src
    assert "_dev" not in src and "AgentConfig(" not in src
    for cell_id in ("70e3b35e", "10185d26023b46108eb7d9f57d49d2b3"):
        assert "{SPLIT}" in "".join(cells[cell_id]["source"])
