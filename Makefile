.PHONY: install test experiment figures all clean

install:
	python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt

test:
	.venv/bin/pytest -q

# Reproduce the reported numbers. On first run this streams the STRING human
# interactome (~79 MB) and the Open Targets Alzheimer gene list, then caches
# them under data/. Subsequent runs take a few seconds.
experiment:
	.venv/bin/python -m src.experiment

figures:
	.venv/bin/python -m src.figures

all: experiment figures

clean:
	rm -rf data __pycache__ src/__pycache__ .pytest_cache
