# Available recipes: `just --list`. Everything here is a thin wrapper around
# `uv`/`uvx`; each recipe's body below is exactly what it runs.
set positional-arguments := true

default:
    @just --list

# (quality) Lint (fails on issues, doesn't fix). Use `fix` to auto-fix.
lint:
    uv run ruff check src
    uv run ruff format --check src

# (quality) Auto-fix lint issues and reformat.
fix:
    uv run ruff check --fix src
    uv run ruff format src

# (annotations) Check static typing.
typecheck:
    uv run ty check

# (test) Run tests. Extra args are forwarded to pytest.
test *args:
    uv run pytest "$@"

# (test) Run tests and show a coverage report (htmlcov/index.html + terminal).
coverage:
    uv run pytest --cov=dw3d --cov-report=html --cov-report=term

# (quality) Run all pre-commit hooks against every file.
precommit:
    uvx pre-commit run --all-files

# (installation) Install pre-commit as a git hook.
precommit-install:
    uvx pre-commit install

# (build) Check that the future sdist/wheel will contain all the files you want.
check-manifest:
    uvx check-manifest -c -v

# (profiling) Run scalene on a script, e.g. `just profile myscript.py`.
profile *args:
    uv run --with scalene scalene "$@"

# (build) Build an SDist and wheel(s) locally, into dist/.
build:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf build dist
    # Pure Python: a single universal wheel, no per-interpreter or per-platform rebuild needed.
    uv build -o dist

# (build) Publish the built packages (needs ~/.pypirc). Pass "pypi" to publish there instead of koda.
publish *args:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ ! -d dist ]]; then
        echo "dist/ not found, have you run 'just build' ?"
        exit 1
    fi
    for f in dist/*.tar.gz dist/*.whl; do
        [[ -e "$f" ]] || continue
        uvx twine check "$f" --strict
    done
    if [[ "$*" == *"pypi"* ]]; then
        uvx twine upload dist/*
    else
        uvx twine upload --repository koda dist/*
    fi

# (clean) Delete cache folders and build artifacts.
clean:
    rm -rf build __pycache__ .pytest_cache .coverage htmlcov .ruff_cache .ty_cache
