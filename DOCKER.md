# Docker setup on Windows

The default image is CPU-compatible and runs on Windows through Docker Desktop
with the WSL 2 backend. NVIDIA GPU support is an optional Compose override.

## Prerequisites

- Docker Desktop using Linux containers and the WSL 2 backend
- Docker Compose
- For GPU execution only: a supported NVIDIA GPU and current Windows driver

Run all commands from the repository root.

## Build the portable CPU image

```powershell
docker compose build
```

Check the installed libraries:

```powershell
docker compose run --rm gradcam3d python -c "import torch, cv2, librosa, mediapipe, shap; print('imports ok'); print('CUDA:', torch.cuda.is_available())"
```

Show the unified analyzer options:

```powershell
docker compose run --rm gradcam3d
```

Run CPU analysis on the included sample:

```powershell
docker compose run --rm gradcam3d python analysis/analyze_video.py /app/small_data/fake/Whatsapp.mp4 --device cpu --transcript-gpu-index -1
```

## Build and use the optional NVIDIA image

First verify that Docker Desktop can access the GPU:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

Build the CUDA 12.1 PyTorch variant:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml build
```

Check CUDA inside the project container:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm gradcam3d python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

Run GPU analysis:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm gradcam3d python analysis/analyze_video.py /app/small_data/fake/Whatsapp.mp4 --device cuda --transcript-gpu-index 0
```

For a 6 GB GPU, start CNN training with a small batch size:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm gradcam3d python cnn/train.py --device cuda --batch-size 2 --num-workers 2
```

## Mounted project data

Datasets, checkpoints, trained models, transcript benchmarks, and outputs are
not embedded in the image. Compose mounts them from the Windows workspace so
that rebuilding the image does not duplicate large files or delete results.

The main container paths are:

- `/app/data`
- `/app/checkpoints`
- `/app/outputs`
- `/app/weak_labels`
- `/app/random_forest/models`
- `/app/transcript/benchmarks`

## Transcript command working directory

The supplied transcript configuration contains paths relative to the
`transcript` directory. Run direct transcript commands with that directory as
the container working directory:

```powershell
docker compose run --rm -w /app/transcript gradcam3d python infer.py compare_transcripts=true data_filename=/app/small_data/fake/Whatsapp.mp4 gpu_idx=-1 detector=mediapipe
```

## Common Windows issues

- Ensure Docker Desktop is running before issuing Docker commands.
- Use Linux containers, not Windows containers.
- If file access is slow, keep the repository inside the WSL filesystem rather
  than under `C:\` for compute-heavy development.
- CPU execution is portable but CNN training, SHAP, and speech inference can be
  slow.
- If GPU memory is exhausted, lower the CNN batch size and SHAP background
  sample count.

## Complete command reference

Run these commands from the repository root:

```powershell
cd "C:\Users\Madhw\Downloads\GradCAM3D - Copy"
```

All paths passed to programs inside a container must use Linux forward slashes.

### Inspect mounted files

```powershell
docker compose run --rm gradcam3d find /app/small_data -maxdepth 3 -type f
docker compose run --rm gradcam3d find /app/checkpoints -maxdepth 2 -type f
docker compose run --rm gradcam3d find /app/data -maxdepth 3 -type f
```

### CPU CNN inference

```powershell
docker compose run --rm gradcam3d python cnn/inference.py /app/small_data/Fake_Video_Fake_Audio.mp4 --checkpoint-path /app/checkpoints/best_model.pth --device cpu
```

### CPU Grad-CAM and head-wise Grad-CAM

```powershell
docker compose run --rm gradcam3d python cnn/gradcam.py /app/small_data/Fake_Video_Fake_Audio.mp4 --checkpoint-path /app/checkpoints/best_model.pth --output-path /app/outputs/gradcam_video.mp4 --device cpu

docker compose run --rm gradcam3d python cnn/headwise_gradcam.py /app/small_data/Fake_Video_Fake_Audio.mp4 --checkpoint-path /app/checkpoints/best_model.pth --target-class 1 --output-path /app/outputs/headwise_fake.mp4 --device cpu
```

### CPU SHAP explanation

```powershell
docker compose run --rm gradcam3d python cnn/shap_explain.py /app/small_data/Fake_Video_Fake_Audio.mp4 --checkpoint-path /app/checkpoints/best_model.pth --background-root /app/data/train --output-path /app/outputs/shap_video.mp4 --npy-path /app/outputs/shap_attribution.npy --raw-npy-path /app/outputs/shap_raw_signed.npy --report-path /app/outputs/shap_report.json --num-background 2 --device cpu
```

### Weak-label generation

```powershell
docker compose run --rm gradcam3d python cnn/generate_weak_labels.py --input-root /app/data/train --weak-label-dir /app/weak_labels/train
```

Add `--overwrite` to replace existing weak-label files.

### CPU CNN training

```powershell
docker compose run --rm gradcam3d python cnn/train.py --train-dir /app/data/train --val-dir /app/data/val --checkpoint-path /app/checkpoints/best_model.pth --device cpu --batch-size 2 --num-workers 0
```

With optional weak concept supervision:

```powershell
docker compose run --rm gradcam3d python cnn/train.py --train-dir /app/data/train --val-dir /app/data/val --weak-label-dir /app/weak_labels/train --checkpoint-path /app/checkpoints/best_model.pth --device cpu --batch-size 2 --num-workers 0
```

### Random Forest feature extraction and training

```powershell
docker compose run --rm gradcam3d python random_forest/extract_features.py --input-csv /app/random_forest/metadata/train.csv --output-csv /app/random_forest/features/train_features.csv

docker compose run --rm gradcam3d python random_forest/extract_features.py --input-csv /app/random_forest/metadata/validation.csv --output-csv /app/random_forest/features/validation_features.csv

docker compose run --rm gradcam3d python random_forest/extract_features.py --input-csv /app/random_forest/metadata/test.csv --output-csv /app/random_forest/features/test_features.csv

docker compose run --rm gradcam3d python random_forest/train.py
```

Run Random Forest prediction:

```powershell
docker compose run --rm gradcam3d python random_forest/predict_video.py /app/small_data/Fake_Video_Fake_Audio.mp4
```

### Transcript comparison

CPU transcript comparison with MediaPipe:

```powershell
docker compose run --rm -w /app/transcript gradcam3d python infer.py compare_transcripts=true data_filename=/app/small_data/Fake_Video_Fake_Audio.mp4 gpu_idx=-1 detector=mediapipe
```

Lower-memory version without semantic similarity:

```powershell
docker compose run --rm -w /app/transcript gradcam3d python infer.py compare_transcripts=true semantic_similarity=false data_filename=/app/small_data/Fake_Video_Fake_Audio.mp4 gpu_idx=-1 detector=mediapipe
```

### Complete CPU analysis

```powershell
docker compose run --rm gradcam3d python analysis/analyze_video.py /app/small_data/Fake_Video_Fake_Audio.mp4 --device cpu --transcript-gpu-index -1
```

Add `--strict` when an incomplete component report should return a failure.

### Rule-based fusion

```powershell
docker compose run --rm gradcam3d python -m rule_based_fusion /app/outputs/full_analysis_Fake_Video_Fake_Audio.json -o /app/outputs/full_analysis_Fake_Video_Fake_Audio_fused.json
```

### GPU commands

Test Docker GPU access:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

Build and verify the GPU image:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml build --progress=plain

docker compose -f compose.yaml -f compose.gpu.yaml run --rm gradcam3d python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

GPU inference and complete analysis:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm gradcam3d python cnn/inference.py /app/small_data/Fake_Video_Fake_Audio.mp4 --checkpoint-path /app/checkpoints/best_model.pth --device cuda

docker compose -f compose.yaml -f compose.gpu.yaml run --rm gradcam3d python analysis/analyze_video.py /app/small_data/Fake_Video_Fake_Audio.mp4 --device cuda --transcript-gpu-index 0
```

GPU CNN training for a 6 GB card:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm gradcam3d python cnn/train.py --train-dir /app/data/train --val-dir /app/data/val --checkpoint-path /app/checkpoints/best_model.pth --device cuda --batch-size 2 --num-workers 2
```

If CUDA memory is exhausted, use `--batch-size 1 --num-workers 0`.

### Help commands

```powershell
docker compose run --rm gradcam3d python analysis/analyze_video.py --help
docker compose run --rm gradcam3d python cnn/inference.py --help
docker compose run --rm gradcam3d python cnn/train.py --help
docker compose run --rm gradcam3d python cnn/gradcam.py --help
docker compose run --rm gradcam3d python cnn/headwise_gradcam.py --help
docker compose run --rm gradcam3d python cnn/shap_explain.py --help
```

### Container and image maintenance

```powershell
docker ps
docker ps -a
docker images
docker compose images
docker compose down
docker compose -f compose.yaml -f compose.gpu.yaml down
```

Optional cache cleanup:

```powershell
docker builder prune
```

Avoid aggressive pruning while actively developing because the next build will
need to download the large dependency layers again.
