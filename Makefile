.PHONY: install data train run test lint docker-up
install:
	pip install -r requirements.txt
data:
	python -m ml.generate_data --out data/sales.csv
train:
	python -m ml.train --data data/sales.csv --out models/
run:
	uvicorn app.main:app --reload
test:
	pytest
lint:
	ruff check .
docker-up:
	docker compose up --build
