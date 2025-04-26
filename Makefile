# Environment name
ENV_NAME = eklipse_pro2
PYTHON_VERSION = 3.10

# =====================
# Commands
# =====================

create-env:
	conda create -y -n $(ENV_NAME) python=$(PYTHON_VERSION)

activate-env:
	@echo "To activate the environment, run:"
	@echo "conda activate $(ENV_NAME)"

clean-env:
	conda remove -y --name $(ENV_NAME) --all

install:
	pip install -r app/requirements.txt

run-app:
	uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# =====================
# Combo commands
# =====================

setup: create-env activate-env