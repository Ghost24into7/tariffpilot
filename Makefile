.PHONY: setup test lint eval gate run demo ui
setup:
	pip install -e ".[dev,api,gemini]"
test:
	pytest -q --cov=tariffpilot --cov-report=term-missing
lint:
	ruff check src tests
eval:
	python -m tariffpilot eval --split dev --report reports/eval_dev.md
gate:
	python -m tariffpilot eval --split test --system both --gate
run:
	uvicorn tariffpilot.api.app:create_app --factory --host 0.0.0.0 --port 8000
ui:
	streamlit run src/tariffpilot/ui/streamlit_app.py
demo:
	python -m tariffpilot classify "Men's cotton knitted t-shirt, short sleeve"
