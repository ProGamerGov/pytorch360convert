# Contributing to pytorch360convert

This project uses simple linting and testing guidelines, as you'll see below.

## Linting


Linting is simple to perform.

```
pip install black flake8 mypy ufmt pytest-cov

```

Linting:

```
cd pytorch360convert
black .
ufmt format .
cd ..
```

Checking:

```
cd pytorch360convert
black --check --diff .
flake8 . --ignore=E203,W503 --max-line-length=88 --exclude build,dist
ufmt check .
mypy . --ignore-missing-imports --allow-redefinition
cd ..
```


## Testing

Tests can run like this:

```
pip install pytest pytest-cov
```

```
cd pytorch360convert
pytest -ra --cov=. --cov-report term-missing
cd ..
```
