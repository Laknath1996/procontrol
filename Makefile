# Variables
PYTHON = PYTHONPATH=. python
SCRIPTS_DIR = scripts

# Default command
all: format lint

# Formatting and Linting
format:
	ruff format .

lint:
	ruff check --fix .

# Experiment Shortcuts
run-fqi:
	$(PYTHON) $(SCRIPTS_DIR)/run_fqi.py
	$(PYTHON) $(SCRIPTS_DIR)/run_fqi.py --include_time

run-sac:
	$(PYTHON) $(SCRIPTS_DIR)/run_sac.py
	$(PYTHON) $(SCRIPTS_DIR)/run_sac.py --include_time

run-ppo:
	$(PYTHON) $(SCRIPTS_DIR)/run_ppo.py
	$(PYTHON) $(SCRIPTS_DIR)/run_ppo.py --include_time

run-plc-nn:
	$(PYTHON) $(SCRIPTS_DIR)/run_plc.py --regressor nn --eval_period 100 --terminal_time 3000

run-plc-rf:
	$(PYTHON) $(SCRIPTS_DIR)/run_plc.py --regressor rf --eval_period 50 --terminal_time 1000

# Clean up cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .ruff_cache