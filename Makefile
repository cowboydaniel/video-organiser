.PHONY: install run-gui run-cli lint format test

VENV?=.venv
PYTHON?=$(VENV)/bin/python
PIP?=$(VENV)/bin/pip

install: $(VENV)/bin/activate
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install black isort mypy pytest typer pydantic PySide6

$(VENV)/bin/activate:
	python -m venv $(VENV)

run-gui:
	PYTHONPATH=src $(PYTHON) -m app.main

run-cli:
	PYTHONPATH=src $(PYTHON) -m tools.cli --help

lint:
	$(PYTHON) -m black --check src
	$(PYTHON) -m isort --check-only src
	$(PYTHON) -m mypy src

format:
	$(PYTHON) -m isort src
	$(PYTHON) -m black src

test:
	$(PYTHON) -m pytest
