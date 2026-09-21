.PHONY: install lint format typecheck test check demo
install:
	python -m venv .venv && .venv/bin/pip install -e ".[dev]"
lint:
	.venv/bin/ruff check .
format:
	.venv/bin/ruff format .
typecheck:
	.venv/bin/mypy
test:
	.venv/bin/pytest
check: lint typecheck test
demo:
	.venv/bin/crowdcast demo
