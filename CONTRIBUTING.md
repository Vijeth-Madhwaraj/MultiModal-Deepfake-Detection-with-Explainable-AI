# Contributing

Thanks for helping improve GradCAM3D. Keep changes focused and avoid committing
datasets, checkpoints, generated videos, or local virtual environments.

## Local setup

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The project contains four largely independent areas:

- `cnn/` — 3D CNN training, inference, and explanations
- `random_forest/` — audio-feature classifier
- `transcript/` — audio/visual speech recognition and comparison
- `Rule_Based_Fusion_Module/` — deterministic result fusion

## Before opening a pull request

Run the checks that apply to the area you changed. At minimum, verify Python
files compile and run the fusion regression tests:

```powershell
python -m compileall cnn random_forest analysis data_tools
python -m unittest discover -s Rule_Based_Fusion_Module\tests -v
```

In the pull request, describe the problem, the approach, the checks you ran,
and any model or dataset assumptions needed to reproduce the result.

## Repository hygiene

- Never commit private or licensed datasets.
- Put generated artifacts in the ignored `outputs/`, `checkpoints/`, or `work/`
  directories.
- Add dependencies to `requirements.txt` only when they are required at runtime.
- Keep labels consistent: `REAL = 0` and `FAKE = 1`.
