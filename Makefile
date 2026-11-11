.PHONY: setup data evaluate test all

setup:
	python -m pip install -r requirements.txt

data:
	python data/generate_corpus.py --seed 42 --out data

evaluate:
	PYTHONPATH=src python -m finrag.pipeline --data data --out reports

test:
	python -m pytest -q

all: data evaluate test
