# Contributing

Install development dependencies with `python -m pip install -e '.[dev]'`.
Before submitting a change, run `ruff check .`, `pytest -q` and `python -m build`.
Prefer small pull requests with a reproducible test and a clear compatibility note.

Changes to a loss, class map, augmentation, model graph or metric require an
explicit protocol note in `docs/reproducibility.md`. Evaluation reports must
come from predictions; distinguish reference-implementation results from paper
results. Preserve upstream model attribution and license notices.

Bug reports should include the commit, command, environment and complete traceback.
Use a small synthetic example where possible and exclude private datasets or tokens.
