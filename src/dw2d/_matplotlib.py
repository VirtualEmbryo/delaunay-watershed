"""matplotlib, imported only when one of dw2d's plotting functions is called.

dw2d's reconstruction and its geometry need NumPy, SciPy, scikit-image and networkx only; matplotlib
draws dw2d's figures and comes with the `viewing` extra (`pip install "delaunay-watershed-2d[viewing]"`).
`pyplot` and `cm` stand for `matplotlib.pyplot` and `matplotlib.cm`: the first attribute read imports
the real module, so `import dw2d` and every reconstruction work without matplotlib, and a plotting call
without it raises an `ImportError` that names the extra.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

VIEWING_EXTRA_HINT = 'install the viewing extra: pip install "delaunay-watershed-2d[viewing]"'


class _ImportedOnFirstUse:
    """Stand-in for a module, imported the first time one of its attributes is read.

    Dunder names (`__wrapped__`, `__file__`, ...) are not delegated: tools such as `inspect`, `doctest`
    or `hasattr` probe them, and must get an `AttributeError`, never an import or an `ImportError`.
    """

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is None:
            try:
                self._module = importlib.import_module(self._module_name)
            except ImportError as error:
                msg = f"{self._module_name} is needed to draw dw2d's figures and is not installed; {VIEWING_EXTRA_HINT}"
                raise ImportError(msg) from error
        return self._module

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - the attributes of an arbitrary module
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._load(), name)

    def __repr__(self) -> str:
        return f"<{self._module_name}, imported on first use>"


pyplot = _ImportedOnFirstUse("matplotlib.pyplot")
cm = _ImportedOnFirstUse("matplotlib.cm")
