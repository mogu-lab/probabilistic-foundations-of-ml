FROM docker.io/deepnote/python:3.10-datascience-ra-54

RUN pip install -r "https://raw.githubusercontent.com/mogu-lab/cs345/master/requirements.txt"

