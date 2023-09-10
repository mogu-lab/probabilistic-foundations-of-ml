build:
	jupyter-book build --all .

clean:
	rm -rf _build

env:
	pip freeze > requirements.txt

style:
	jupyter-book config sphinx .
