SPS Studio (SPS_Studio)
=========================

Lightweight desktop app for statistical analysis and SPC workflows.

Quick setup
-----------
1. Create a virtual environment (recommended):

   python -m venv .venv
   .\.venv\Scripts\activate

2. Install requirements:

   python -m pip install -r requirements.txt

Building the documentation
--------------------------
From the repository root run:

```powershell
python -m pip install -r requirements.txt
python -m sphinx -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` after the build completes.

Notes
-----
- This repository contains PySide6 UI code and numerical modules using numpy/pandas/scipy.
- The `docs/` folder contains Sphinx configuration and the API autosummary skeleton.
