#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

echo "safety (failure is tolerated)"
safety check -r requirements/prod.txt -r requirements/dev.txt

echo "pylint"
pylint few_shot_image_classification training || FAILURE=true

echo "pycodestyle"
pycodestyle few_shot_image_classification training || FAILURE=true

echo "pydocstyle"
pydocstyle few_shot_image_classification training || FAILURE=true

echo "mypy"
mypy few_shot_image_classification training || FAILURE=true

echo "bandit"
bandit -ll -r {few_shot_image_classification,training} || FAILURE=true

echo "shellcheck"
find . -name "*.sh" -print0 | xargs -0 shellcheck || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 1
fi
echo "Linting passed"
exit 0