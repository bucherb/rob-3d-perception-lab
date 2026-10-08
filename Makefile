.PHONY: help test test-shared test-lab1 lint clean check-submission check-env

LAB ?= 1

help:
	@echo "make check-env         check the environment (add GPU=1 on a GPU node)"
	@echo "make test              run every contract test"
	@echo "make test-shared       run the rob498 package tests (must pass from day one)"
	@echo "make test-lab1         run Lab 1 contract tests"
	@echo "make check-submission  check deliverables before you zip"
	@echo "make clean             remove caches"

test:
	pytest rob498/tests lab1/tests -v

test-shared:
	pytest rob498/tests -v
test-lab1:
	pytest lab1/tests -v

check-env:
	python scripts/check_env.py $(if $(GPU),--gpu,)

lint:
	python -m compileall -q rob498 lab1

# Checks that the required files exist and parse. It does NOT check that your
# numbers are right -- it checks you will not lose points on file format.
check-submission:
	@python scripts/check_submission.py --lab $(LAB)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
