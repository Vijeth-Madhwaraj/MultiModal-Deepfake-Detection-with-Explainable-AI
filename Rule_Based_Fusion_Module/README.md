# Rule-Based Fusion Module

This repository implements an explainable, deterministic fusion layer for deepfake
detection. Its standalone input is a JSON report containing results from a 3D CNN,
weak-label heuristics, a random forest, and transcript comparison. It validates
those results and appends a four-way audio/video decision at
`results.rule_based_fusion`.

The repository does not implement or train the upstream detection models. An MP4
path may be recorded in the JSON as metadata, but the fusion engine operates on the
model-result JSON itself.

## Repository structure

```text
Rule_Based_Fusion_Module/
|-- config/                  # thresholds and weights
|-- examples/                # valid sample input and clean template
|-- output/                  # generated fused reports
|-- schemas/                 # formal JSON contracts
|-- src/rule_based_fusion/   # Python package
|-- tests/                   # automated regression tests
|-- pyproject.toml           # package and CLI definition
`-- README.md
```

## Input format

Copy `examples/input_template.json` and replace its values with actual upstream
model results. The authoritative machine-readable contract is
`schemas/detector_report.schema.json`.

Required result sections:

```text
results
|-- cnn_3d
|-- weak_label_heuristics
|-- random_forest
`-- transcript_comparison
```

Probabilities and heuristic scores use `[0, 1]`. Transcript percentage metrics use
`[0, 100]`. Labels are `REAL` (`0`) and `FAKE` (`1`). Random-forest probabilities
must sum to one, and its confidence must equal the probability of its selected
label. The report status must be `complete`.

## Decision rules

The module makes separate visual and audio decisions before combining them:

1. **Video authenticity** uses the 3D CNN and visual weak-label heuristics
   (boundary inconsistency and eye irregularity). Their configured weighted
   average is compared with `video_fake_score`.
2. **Audio authenticity** uses the Random Forest fake probability and compares
   it with `audio_fake_score`.
3. **Cross-modal consistency** uses transcript mismatch evidence. It is reported
   separately because an audio/visual transcript disagreement cannot, by itself,
   identify which modality is fake.
4. **Final result** combines the video and audio decisions into one of four
   labels:

| Label | ID | Meaning |
|---|---:|---|
| `REAL_VIDEO_REAL_AUDIO` | 0 | Both modalities are classified as real |
| `REAL_VIDEO_FAKE_AUDIO` | 1 | Video is real and audio is fake |
| `FAKE_VIDEO_REAL_AUDIO` | 2 | Video is fake and audio is real |
| `FAKE_VIDEO_FAKE_AUDIO` | 3 | Both modalities are classified as fake |

The final confidence is the lower of the video and audio decision confidences,
so a weak modality cannot be hidden by a very confident one.

Default visual source weights:

| Source | Weight |
|---|---:|
| 3D CNN | 2/3 |
| Weak-label heuristics | 1/3 |

All thresholds and weights are in `config/default_rules.json`. The formulas and
rule priority are implemented in `src/rule_based_fusion/engine.py`.

## Install

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

If an existing `.venv` points to a removed Python installation, recreate it using
a working Python installation.

## Validate a detector report

```powershell
rule-based-fusion examples\00181_input.json --validate-only
```

Expected output:

```text
Detector JSON is valid.
```

## Run fusion

```powershell
rule-based-fusion examples\00181_input.json -o output\00181_fused.json
```

The original report is preserved. The generated decision is added under
`results.rule_based_fusion`. It contains the four-way `label`, `label_id`, and
`confidence`, plus detailed `video`, `audio`, `cross_modal_consistency`,
`evidence`, and `rule_trace` objects. Its contract is documented in
`schemas/rule_based_fusion_result.schema.json`.

## Run with the GradCAM3D project

The parent GradCAM3D project already produces the required detector JSON. From
the parent project root, expose this package's `src` directory for the current
PowerShell session:

```powershell
$env:PYTHONPATH="$PWD\Rule_Based_Fusion_Module\src"
```

Fuse an existing complete analysis report:

```powershell
python -m rule_based_fusion outputs\full_analysis_Whatsapp.json -o outputs\full_analysis_Whatsapp_fused.json
```

Or run the upstream analysis and fusion from one MP4 command:

```powershell
python -m rule_based_fusion small_data\fake\Whatsapp.mp4 `
  --inference-config Rule_Based_Fusion_Module\config\gradcam3d_inference.json `
  -o outputs\Whatsapp_fused.json
```

The included integration configuration uses `best_model_2.pth`, runs the CNN
on CUDA, runs transcript inference on CPU, requires every upstream stage to
succeed, and allows up to two hours. Run these commands from the parent project
root because the configuration intentionally uses project-relative paths.

Use a custom rule configuration when needed:

```powershell
rule-based-fusion examples\00181_input.json `
  --config config\default_rules.json `
  -o output\00181_fused.json
```

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Optional future MP4 integration

`src/rule_based_fusion/pipeline.py` and `config/inference.example.json` provide an
optional adapter for later integration with the working upstream models. The
adapter can invoke an external inference program, validate its generated detector
JSON, and pass that JSON to the same unchanged fusion engine. It does not infer
model scores by itself or derive labels from filenames.
