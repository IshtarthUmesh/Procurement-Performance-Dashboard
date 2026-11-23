# Contributing

Thanks for helping improve the Procurement Spend Analysis Dashboard.

## Local setup

1. Fork and clone the repository.
2. Create a Python 3.11+ virtual environment.
3. Install development dependencies with `python -m pip install -r requirements-dev.txt`.
4. Run the dashboard with `streamlit run app.py`.

## Before opening a pull request

Run the same checks as CI:

```bash
python -m ruff check .
python -m compileall -q app.py src views scripts tests
python -m pytest -q
```

Keep analytics logic in `src/` as pure, testable functions. Add a regression test for every
bug fix and document any new savings assumption in both `src/config.py` and the README.
Do not commit confidential procurement data, credentials, or Streamlit secrets.

## Pull requests

Use a focused title, explain the business impact, and include screenshots for visible UI
changes. Generated data and the model must remain reproducible from the scripts in `scripts/`.
