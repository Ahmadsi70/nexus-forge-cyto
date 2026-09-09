# Deterministic gate commands for no-mistakes (avoid agent CLI argv overflow on Windows).
.PHONY: test lint

test:
	python -m pytest tests/ -q --ignore=tests/validation

lint:
	python -m compileall -q services scripts
	cargo check --quiet
