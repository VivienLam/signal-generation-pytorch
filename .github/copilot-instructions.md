<!-- Auto-generated guidance for AI coding agents working on this repo -->
# Copilot / AI agent instructions — signal-generation-pytorch

Purpose
- Help contributors quickly understand the project's structure, common workflows, and concrete examples to make small, safe code changes.

Big picture
- This repository is a small PyTorch-based signal generation project. Key components are:
  - `notebooks/` — primary exploratory code (data generation example: `notebooks/01_data_generation.ipynb`).
  - `test_torch.py` — tiny smoke test that verifies PyTorch is functional (prints a gradient).
  - `requirements.txt` — lists direct Python dependencies: `torch`, `numpy`, `matplotlib`, `scikit-learn`, `jupyter`.
  - `data/`, `models/`, `training/`, `utils/` — present but currently empty; expect scripts and outputs to be added here.

What to look for when editing
- Preserve the project's minimal, exploratory style: many examples live in notebooks rather than production scripts.
- Keep changes small and focused: update or add a script in `training/` or `utils/` rather than changing notebooks unless improving reproducibility.

Developer workflows (discoverable)
- Setup environment:

```bash
python -m pip install -r requirements.txt
```

- Quick smoke test (verifies PyTorch + autograd):

```bash
python test_torch.py
# Expected output includes: "Gradient :" followed by a tensor
```

- Open the data-generation notebook to inspect and run examples:

```bash
jupyter notebook notebooks/01_data_generation.ipynb
```

Project-specific conventions
- Notebooks contain primary examples and exploratory code — prefer adding small utility scripts under `utils/` and training scripts under `training/` for reusable functionality.
- Models and datasets should be placed under `models/` and `data/` respectively when produced by scripts (these folders exist intentionally).

Integration points & dependencies
- External libs are the ones in `requirements.txt`. There are no discovered cloud, DB, or remote APIs in the repo — changes can assume local file I/O.

How to make safe changes
- When adding code that will be invoked outside notebooks, add a small runnable script and a minimal smoke test similar to `test_torch.py`.
- Do not modify notebooks programmatically; instead extract reusable code into `utils/` and reference it from the notebook.

Files to inspect for examples
- `README.md` — repo title and brief description.
- `test_torch.py` — example of a minimal, verifiable script.
- `notebooks/01_data_generation.ipynb` — canonical example for data generation and experiments.

If anything is unclear or you need more detailed patterns (example training scripts, model saving conventions, or CI commands), ask the maintainer and I'll expand this guidance.
