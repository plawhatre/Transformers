SHELL := /bin/bash

venv:
	python3 -m venv transformer
	source ./transformer/bin/activate
	pip install -r requirements.txt