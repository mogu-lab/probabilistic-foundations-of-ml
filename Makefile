build:
	jupyter-book build --all .

clean:
	rm -rf _build && rm -rf __pycache__

env:
	pip freeze > requirements.txt

style:
	jupyter-book config sphinx .
