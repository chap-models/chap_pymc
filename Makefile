.PHONY: lint format check test coverage clean install dev

# Install dependencies
install:
	uv sync

# Install with dev dependencies
dev:
	uv sync --all-extras

# Run all linting and type checking
lint: format check

# Format code with ruff
format:
	uv run ruff format chap_pymc tests main.py
	uv run ruff check --fix chap_pymc tests main.py

# Run type checkers (without fixing)
check:
	uv run ruff check chap_pymc tests main.py
	uv run mypy chap_pymc main.py

# Run tests
test:
	uv run pytest tests -v -m 'not slow'

# Run tests with coverage
coverage:
	uv run pytest tests -v --cov=chap_pymc --cov-report=term-missing --cov-report=html -m 'not slow'

# Clean build artifacts
clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
