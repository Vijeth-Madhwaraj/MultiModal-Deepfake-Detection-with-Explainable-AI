# GradCAM3D

> Explainable multimodal deepfake detection with a 3D CNN, audio analysis,
> transcript consistency, and deterministic decision fusion.

PyTorch project for binary deepfake video classification (`Real` vs `Fake`) using a custom 3D CNN, face-focused video clips, Grad-CAM++, head-wise Grad-CAM++, and SHAP explanations.

For a detailed explanation of every active subsystem and the complete data
flow, see [`PROJECT_WORKFLOW.md`](PROJECT_WORKFLOW.md).

## Highlights

- Face-focused 3D CNN classification with Grad-CAM++ and SHAP explanations
- Audio classification using engineered features and a Random Forest
- Audio/video transcript comparison with word, character, and semantic metrics
- Explainable four-way fusion of video and audio authenticity decisions

## Project layout

```text
cnn/                 3D CNN model, training, inference, and explanations
random_forest/       Audio features, Random Forest training, and prediction
transcript/          Audio/visual speech recognition and transcript metrics
data_tools/          Dataset preparation and split utilities
analysis/            Unified multimodal analysis entry point
Rule_Based_Fusion_Module/  Deterministic decision fusion package
data/                Train, validation, and test videos
checkpoints/         Trained CNN checkpoints
outputs/             Generated CNN explanation videos and arrays
weak_labels/         Optional semantic weak labels for CNN training
```

The runnable systems are deliberately separated. Run the CNN, Random Forest,
and data-tool commands below from the repository root. Run transcript commands
from inside `transcript/` because its downloaded model configuration uses paths
relative to that folder.

### Unified JSON analysis

Run every detector and transcript metric for one video and save one report:

```powershell
python analysis\analyze_video.py small_data\fake\Whatsapp.mp4 --device cpu --transcript-gpu-index -1
```

The default output is `outputs/full_analysis_<video-name>.json`. Labels use
`REAL = 0` and `FAKE = 1`. The report contains 3D CNN classification and
confidence, weak-label heuristic scores, Random Forest classification and
probabilities, audio/video transcripts, WER, word-match rate, character
similarity, and semantic similarity. Add `--strict` when an incomplete report
should return a non-zero exit code.

### Rule-based fusion

Fuse an existing complete analysis JSON from the project root:

```powershell
$env:PYTHONPATH="$PWD\Rule_Based_Fusion_Module\src"
python -m rule_based_fusion outputs\full_analysis_Whatsapp.json -o outputs\full_analysis_Whatsapp_fused.json
```

To run analysis and fusion together directly from an MP4:

```powershell
python -m rule_based_fusion small_data\fake\Whatsapp.mp4 --inference-config Rule_Based_Fusion_Module\config\gradcam3d_inference.json -o outputs\Whatsapp_fused.json
```

The fusion result is one of `REAL_VIDEO_REAL_AUDIO`,
`REAL_VIDEO_FAKE_AUDIO`, `FAKE_VIDEO_REAL_AUDIO`, or
`FAKE_VIDEO_FAKE_AUDIO`. The JSON also reports separate video, audio, and
audio/visual transcript-consistency decisions.

### Random Forest and transcript tools

- Random Forest workflow: see [`random_forest/README.md`](random_forest/README.md).
- Transcript/visual-speech workflow: see [`transcript/README.md`](transcript/README.md).

Typical entry points are:

```powershell
python random_forest\train.py
python random_forest\predict_video.py "C:\path\to\video.mp4"
Push-Location transcript
python infer.py config_filename=configs/LRS3_V_WER19.1.ini data_filename="C:\path\to\video.mp4"
Pop-Location
```

## Requirements

- Windows PowerShell (the commands below are written for Windows)
- Python 3.10, 3.11, or 3.12
- An NVIDIA GPU is optional; CPU mode is supported but training and SHAP will be slow
- Celeb-DF/Celeb-DF-v2, or another dataset arranged as shown below

## Docker on Windows

The project includes a portable CPU image and an optional NVIDIA GPU variant.
Docker Desktop must use Linux containers with the WSL 2 backend.

Build and verify the CPU image:

```powershell
docker compose build --progress=plain
docker compose run --rm gradcam3d python -c "import torch, cv2, librosa, mediapipe, shap; print('Imports OK'); print('CUDA:', torch.cuda.is_available())"
```

Run CPU inference:

```powershell
docker compose run --rm gradcam3d python cnn/inference.py /app/small_data/Fake_Video_Fake_Audio.mp4 --checkpoint-path /app/checkpoints/best_model.pth --device cpu
```

Build and verify the optional NVIDIA image:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml build --progress=plain
docker compose -f compose.yaml -f compose.gpu.yaml run --rm gradcam3d python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Container paths use Linux forward slashes (`/`), even when commands are issued
from Windows PowerShell. See [`DOCKER.md`](DOCKER.md) for the complete Windows
command reference covering inference, training, Grad-CAM, SHAP, transcripts,
Random Forest processing, fusion, mounted folders, and cleanup.

Run every command from the project root (`GradCAM3D`). Do not reuse a copied `.venv` from another computer or Python installation; virtual environments contain machine-specific paths.

## 1. Create the environment and install dependencies

```powershell
py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py -3.10` is unavailable, install Python 3.10-3.12 and replace `py -3.10` with the installed launcher version.

Check the installation and detected device:

```powershell
python -c "import torch, cv2, numpy, shap; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('OpenCV:', cv2.__version__)"
```

`requirements.txt` installs the CUDA 12.1 PyTorch wheel when it is available. To explicitly reinstall it for an NVIDIA GPU:

```powershell
python -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

If the machine does not have a compatible NVIDIA GPU, use `--device cpu` in the commands below.

## 2. Prepare the dataset

The final directory layout must be:

```text
data/
|-- train/
|   |-- real/
|   `-- fake/
|-- val/
|   |-- real/
|   `-- fake/
`-- test/
    |-- real/
    `-- fake/
```

Supported video extensions are `.mp4`, `.avi`, `.mov`, `.mkv`, and `.webm`.

If the data is already arranged this way, skip to training. To split raw Celeb-DF folders, replace the example paths with the real locations:

```powershell
python data_tools\prepare_dataset.py --sources "C:\path\to\Celeb-DF" "C:\path\to\Celeb-DF-v2" --output-dir data --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42 --manifest-path data\split_manifest.csv
```

Important: `data_tools/prepare_dataset.py` deletes and rebuilds the existing `data/train`, `data/val`, and `data/test` directories. By default it copies the source videos and balances real/fake classes by downsampling the larger class. Add `--move` only if you intentionally want to remove videos from their original locations.

Check how many videos were prepared:

```powershell
Get-ChildItem data\train\real -File | Measure-Object
Get-ChildItem data\train\fake -File | Measure-Object
Get-ChildItem data\val\real -File | Measure-Object
Get-ChildItem data\val\fake -File | Measure-Object
Get-ChildItem data\test\real -File | Measure-Object
Get-ChildItem data\test\fake -File | Measure-Object
```

## 3. Train the model

GPU training with the code defaults (20 epochs, batch size 16, learning rate `3e-4`, four data-loader workers):

```powershell
python cnn\train.py --device cuda
```

CPU training, with conservative settings:

```powershell
python cnn\train.py --device cpu --batch-size 2 --num-workers 0
```

Full example with the most useful options:

```powershell
python cnn\train.py --train-dir data\train --val-dir data\val --epochs 20 --batch-size 16 --lr 3e-4 --num-workers 4 --device cuda --checkpoint-path checkpoints\best_model.pth
```

Optional training modes:

```powershell
# Use per-video weak labels during training
python cnn\train.py --device cuda --weak-label-dir weak_labels\train

# Initialize compatible backbone weights from an older checkpoint
python cnn\train.py --device cuda --pretrained-backbone-path checkpoints\old_model.pth

# Enable landmark-based eye alignment
python cnn\train.py --device cuda --align-faces

# Disable the ReduceLROnPlateau scheduler
python cnn\train.py --device cuda --no-scheduler
```

Weak labels are optional. Without `--weak-label-dir`, training uses the Real/Fake class labels and cross-entropy loss only. When `--weak-label-dir` is supplied, each training video must have a matching weak-label JSON or NPY file and the semantic concept loss is added with the configured weight of `0.3`.

The best validation model is saved to `checkpoints/best_model.pth`. Lower `--batch-size` if GPU memory is exhausted. On Windows, use `--num-workers 0` if worker processes cause loading errors.

## 4. Run inference

Use the included sample video and checkpoint:

```powershell
python cnn\inference.py small_data\fake\Whatsapp.mp4 --checkpoint-path checkpoints\best_model.pth --device cuda
```

For another video or CPU execution:

```powershell
python cnn\inference.py "C:\path\to\video.mp4" --checkpoint-path checkpoints\best_model.pth --device cpu
```

Add `--align-faces` only when the checkpoint was trained with the same face-alignment option.

## 5. Generate Grad-CAM++ output

Generate the standard explanation video:

```powershell
python cnn\gradcam.py small_data\fake\Whatsapp.mp4 --checkpoint-path checkpoints\best_model.pth --output-path outputs\gradcam_video.mp4 --device cuda
```

The default target layer is `final_conv`. It can be selected explicitly:

```powershell
python cnn\gradcam.py "C:\path\to\video.mp4" --target-layer final_conv --output-path outputs\gradcam_video.mp4 --device cpu
```

## 6. Generate head-wise Grad-CAM++ output

This produces a grid showing the semantic attention heads:

```powershell
python cnn\headwise_gradcam.py small_data\fake\Whatsapp.mp4 --checkpoint-path checkpoints\best_model.pth --output-path outputs\headwise_gradcam_grid.mp4 --device cuda
```

To explain a specific class, use `--target-class 0` for Real or `--target-class 1` for Fake. If omitted, the predicted class is explained.

```powershell
python cnn\headwise_gradcam.py "C:\path\to\video.mp4" --target-class 1 --alpha 0.4 --frame-size 112 --clip-length 16 --device cpu
```

## 7. Generate SHAP output

SHAP is computationally expensive. Start with a small background set:

```powershell
python cnn\shap_explain.py small_data\fake\Whatsapp.mp4 --checkpoint-path checkpoints\best_model.pth --background-root data\train --output-path outputs\shap_video.mp4 --npy-path outputs\shap_attribution.npy --raw-npy-path outputs\shap_raw_signed.npy --report-path outputs\shap_report.json --num-background 2 --device cuda
```

Explain the Fake class logit (`0` is Real, `1` is Fake):

```powershell
python cnn\shap_explain.py "C:\path\to\video.mp4" --target-family class --target-index 1 --output-kind logit --background-root data\train --device cuda
```

Explain a named semantic concept:

```powershell
python cnn\shap_explain.py "C:\path\to\video.mp4" --target-family concept --concept-name boundary_inconsistency --output-kind score --background-root data\train --device cuda
```

Available configured concepts are `boundary_inconsistency` and `eye_blink_irregularity`. Reduce `--num-background` if SHAP runs out of memory. SHAP saves both an overlay video and an aggregated `.npy` attribution volume.

The signed SHAP video uses red for evidence supporting the explained target and blue for evidence opposing it. The JSON report records the reference value, total positive support, total negative opposition, approximation residual, final target value, and signed contribution from every sampled frame. The raw NPY retains the original signed channel-level SHAP values.

## Quick run using the included files

After installing dependencies, the repository already contains a checkpoint and a fake sample video, so the shortest test sequence is:

```powershell
.\.venv\Scripts\Activate.ps1
python cnn\inference.py small_data\fake\Whatsapp.mp4 --device cpu
python cnn\gradcam.py small_data\fake\Whatsapp.mp4 --output-path outputs\gradcam_video.mp4 --device cpu
python cnn\headwise_gradcam.py small_data\fake\Whatsapp.mp4 --output-path outputs\headwise_gradcam_grid.mp4 --device cpu
```

## Command help

The source of truth for every optional argument is the built-in help:

```powershell
python data_tools\prepare_dataset.py --help
python cnn\train.py --help
python cnn\inference.py --help
python cnn\gradcam.py --help
python cnn\headwise_gradcam.py --help
python cnn\shap_explain.py --help
```

## Outputs

- `checkpoints/best_model.pth` - best training checkpoint
- `outputs/gradcam_video.mp4` - standard Grad-CAM++ overlay
- `outputs/headwise_gradcam_grid.mp4` - per-head Grad-CAM++ grid
- `outputs/shap_video.mp4` - SHAP overlay
- `outputs/shap_attribution.npy` - SHAP attribution array

## Troubleshooting

- **The virtual environment cannot create a process:** delete the non-portable `.venv` directory and repeat step 1.
- **CUDA was requested but is unavailable:** install the CUDA wheel shown above or change `--device cuda` to `--device cpu`.
- **CUDA out of memory:** lower `--batch-size`; for SHAP also lower `--num-background`.
- **No training samples found:** verify the exact `train/real`, `train/fake`, `val/real`, and `val/fake` layout.
- **Checkpoint size mismatch:** use the checkpoint produced by this model version, or initialize training with `--pretrained-backbone-path` for compatible older weights.
- **PowerShell blocks activation:** run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.

## Contributing and license

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup,
testing, and repository-hygiene guidance. This project is distributed under the
terms in [`LICENSE`](LICENSE).
