# Contributing

## Development Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Project Standards

- Keep pipeline components modular (`registration`, `segmentation`, `change_detection`, `visualization`).
- Expose new parameters through `configs/default.yaml`.
- Update `README.md` and `docs/method.md` when method behavior changes.

## Pull Request Checklist

- [ ] Code compiles (`python -m compileall src run_pipeline.py`)
- [ ] New params documented in config section
- [ ] Output files and metrics remain reproducible
- [ ] Any model changes include rationale and tradeoffs
