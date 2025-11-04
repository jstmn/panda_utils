#!/bin/bash

black src/**/*.py --line-length 120
black tests/*.py --line-length 120
black scripts/**/*.py --line-length 120

ruff check src/**/*.py --fix
ruff check tests/*.py --fix
ruff check scripts/**/*.py --fix