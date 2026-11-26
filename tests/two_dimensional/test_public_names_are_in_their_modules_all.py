"""Every public name a front end uses is listed in its module's `__all__`.

napari-foambryo's tests use `dw2d.io.load_rec` and `save_rec`; the module had no `__all__`. It now lists every public
name it defines (its own functions, classes and constants, not the names it imports).
"""

from __future__ import annotations

import ast
import importlib
import inspect

import pytest

USED_OUTSIDE = {
    "dw2d.io": ["load_rec", "save_rec"],
}


def _public_names_defined_in(module: object) -> set[str]:
    names = set()
    for node in ast.parse(inspect.getsource(module)).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not name.startswith("_")} - {"__all__"}


@pytest.mark.parametrize(("module_name", "names"), sorted(USED_OUTSIDE.items()))
def test_the_names_used_outside_are_in_a_complete_all(module_name, names):
    module = importlib.import_module(module_name)
    assert set(names) <= set(module.__all__)
    assert set(module.__all__) == _public_names_defined_in(module)
    namespace: dict = {}
    exec(f"from {module_name} import *", namespace)  # noqa: S102 - the star import is what __all__ decides
    assert set(namespace) - {"__builtins__"} == set(module.__all__)
