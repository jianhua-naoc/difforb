PYTHON ?= python

.PHONY: docs-install docs-build docs-serve

docs-install:
	$(PYTHON) -m pip install -r docs/requirements.txt

docs-build:
	$(PYTHON) -m mkdocs build --strict

docs-serve:
	$(PYTHON) -m mkdocs serve
