#!/bin/bash

# Black
python -m black src/*.py --line-length 120
python -m black src/**/*.py --line-length 120
python -m black scripts/*.py --line-length 120
python -m black scripts/**/*.py --line-length 120

# Ruff
python -m  ruff  check  scripts/*.py --fix
python -m  ruff  check  scripts/**/*.py --fix
python -m  ruff  check  src/*.py --fix
python -m  ruff  check  src/**/*.py --fix

