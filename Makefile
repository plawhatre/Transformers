SHELL := /bin/bash

venv:
	python3 -m venv transformer
	source ./transformer/bin/activate
	pip install -r requirements.txt
	
download:
	wget -O data.zip "https://ai4b-my.sharepoint.com/:u:/g/personal/sumanthdoddapaneni_ai4bharat_org/EXhX84sbTQhLrsURCU9DlUwBVyJ10cYK9bQQe1SMljf_yA?e=q7GJpb&download=1"
	unzip data.zip -d data
