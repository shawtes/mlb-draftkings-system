.PHONY: dev build train backtest clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

dev: ## Start web app (frontend + backend)
	cd web_optimizer && npm run dev

server: ## Start backend only
	cd web_optimizer && node server/index.js

client: ## Start frontend only
	cd web_optimizer/client && npm run dev

build: ## Production build of frontend
	cd web_optimizer/client && npx vite build

train: ## Run MLB training pipeline
	python training/mlb/training.py --data-path data/merged_fangraphs_data.csv --output-dir training/mlb/output

train-fast: ## Run training without HPO
	python training/mlb/training.py --data-path data/merged_fangraphs_data.csv --output-dir training/mlb/output --skip-hpo

backtest: ## Run backtesting
	python pipeline/backtest_optimizer.py

pipeline: ## Run full fetch-train-bridge pipeline
	python pipeline/run_pipeline.py

clean: ## Remove generated files
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf web_optimizer/client/build
