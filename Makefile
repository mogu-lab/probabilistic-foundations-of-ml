build:
	python render_schedule.py > schedule.md && jupyter-book build --all .

plots:
	python plots.py

clean:
	rm -rf _build && rm -rf __pycache__

env:
	pip freeze > requirements.txt

style:
	jupyter-book config sphinx .
