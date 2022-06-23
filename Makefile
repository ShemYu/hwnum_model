.PHONY: test

check_dirs := src tests

test:
	python -m pytest

check:
	black --check $(check_dirs)
	isort --check-only .

fix:
	black $(check_dirs)
	isort .