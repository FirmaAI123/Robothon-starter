.PHONY: install run demo collect test info clean

install:
	python -m pip install -r requirements.txt

run:            ## headless autonomous task, prints stage pass/fail
	python run.py autonomous

demo:           ## render the annotated demo video
	python run.py record demo.mp4

collect:        ## generate a 5-episode randomized dataset
	python run.py collect dataset 5

test:
	pytest -q tests/

info:
	python run.py info

clean:
	rm -rf dataset demo.mp4 __pycache__ dexsuite/__pycache__ tests/__pycache__ .pytest_cache
