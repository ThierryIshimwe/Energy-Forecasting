# ─────────────────────────────────────────────────────────────────────
#  Energy Forecasting — one-command operations
#  Usage: `make <target>`. Run `make help` for the full list.
# ─────────────────────────────────────────────────────────────────────

PYTHON := python
PIP := $(PYTHON) -m pip

.PHONY: help setup data eda features tune train evaluate report report-pdf report-html slides app test test-fast lint format type-check leakage-check clean clean-cache clean-all

help:  ## Show this help message
	@echo ""
	@echo "Energy Forecasting — Make targets:"
	@echo ""
	@echo "  Setup & data"
	@echo "    setup            Install package + dev dependencies + pre-commit hooks"
	@echo "    data             Download UCI dataset and verify integrity"
	@echo ""
	@echo "  Pipeline (run in order)"
	@echo "    eda              Execute the EDA notebook"
	@echo "    features         Build feature matrix from cleaned data"
	@echo "    tune             Hyperparameter optimization for all models"
	@echo "    train            Train all models with best hyperparameters"
	@echo "    evaluate         Generate leaderboard and per-fold metrics"
	@echo ""
	@echo "  Reporting & app"
	@echo "    report           Build technical report (PDF) and slides"
	@echo "    report-pdf       Only the technical report PDF"
	@echo "    report-html      HTML version of the report"
	@echo "    slides           Executive summary slides"
	@echo "    app              Launch Streamlit dashboard"
	@echo ""
	@echo "  Quality gates"
	@echo "    test             Full test suite with coverage"
	@echo "    test-fast        Skip slow + integration tests"
	@echo "    lint             flake8 + mypy (read-only checks)"
	@echo "    format           black + isort (writes changes)"
	@echo "    type-check       mypy only"
	@echo "    leakage-check    Run leakage-prevention integration tests"
	@echo ""
	@echo "  Cleanup"
	@echo "    clean            Remove build artifacts (keeps caches)"
	@echo "    clean-cache      Remove Python and tool caches"
	@echo "    clean-all        Both of the above"
	@echo ""

# ── Setup & data ────────────────────────────────────────────────────
setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install

data:
	$(PYTHON) scripts/download_data.py

# ── Pipeline ────────────────────────────────────────────────────────
eda:
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

features:
	$(PYTHON) scripts/build_features.py

tune:
	$(PYTHON) scripts/tune.py

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

# ── Reporting ───────────────────────────────────────────────────────
report: report-pdf slides

report-pdf:
	cd reports/source && quarto render technical_report.qmd --to pdf

report-html:
	cd reports/source && quarto render technical_report.qmd --to html

slides:
	$(PYTHON) scripts/generate_slides.py

app:
	streamlit run app/app.py

# ── Quality gates ───────────────────────────────────────────────────
test:
	pytest

test-fast:
	pytest -m "not slow and not integration"

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

type-check:
	mypy src/

leakage-check:
	pytest -m leakage -v

# ── Cleanup ─────────────────────────────────────────────────────────
clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info

clean-cache:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/

clean-all: clean clean-cache
