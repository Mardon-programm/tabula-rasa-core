# Contributing to TR-Core

Thank you for contributing to TR-Core! This document outlines the process for contributing to the project.

## 🚀 Quick Start

```bash
# Fork the repo, then:
git clone https://github.com/your-username/tabula-rasa-core.git
cd tabula-rasa-core

# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install in development mode
uv pip install -e .[dev]

# Run tests to verify setup
pytest tests/ -v
```

## 📋 Development Workflow

1. **Pick an issue** or create a new one
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make changes** with tests
4. **Run quality checks**:
   ```bash
   ruff check .          # lint
   ruff format --check . # format
   mypy behavioral/ environment/ experiments/
   pytest tests/ -v
   ```
5. **Commit**: `git commit -m "feat: add amazing feature"`
6. **Push & PR**: `git push origin feature/your-feature-name`

## 📝 Code Style

- **Formatter**: `ruff format` (100 char line length)
- **Linter**: `ruff check` (E, F, I, UP, B, C4, ANN, T20)
- **Types**: `mypy` (strict mode where possible)
- **Imports**: Absolute imports, sorted by ruff

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=behavioral --cov=environment --cov=experiments

# Run specific experiment
python -m experiments.run --experiment phase4 --seeds 1-3 --summarize
```

## 📝 Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add GNN baseline to compositional_baselines
fix: fix binomial filter edge case in develop()
docs: update API docs for SymbolicInterface
refactor: simplify SlotAttention module
test: add unit tests for AttributeHead
```

## 🔬 Experiment Guidelines

- All experiments must be reproducible (fixed seeds, Hydra configs)
- New baselines go in `experiments/compositional_baselines.py`
- Results tracked via DVC (`dvc repro`)
- Statistical rigor: permutation tests (20k), Cohen's dz, bootstrap CI

## 📚 Documentation

- Update docstrings for public APIs
- Update README.md for user-facing changes
- Update ARCHITECTURE.md for architectural changes

## 🤝 Code Review

All PRs require:
- ✅ CI passes (lint, typecheck, tests)
- ✅ At least one approval
- ✅ No decrease in test coverage
- ✅ Updated docs if user-facing

## 📞 Getting Help

- **Questions**: Open a Discussion
- **Bugs**: Use the Bug Report template
- **Features**: Use the Feature Request template
- **Security**: Email security@ (see SECURITY.md)

---

*Thank you for contributing to TR-Core!* 🧠
