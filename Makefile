build:
	jupyter-book build --all .

clean:
	rm -rf _build

env:
	conda env export --no-builds | grep -v "^prefix: " > environment.yml
