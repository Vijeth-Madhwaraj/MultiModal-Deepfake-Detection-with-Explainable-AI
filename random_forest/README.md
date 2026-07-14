# Random Forest audio classifier

This folder is self-contained: audio preprocessing, handcrafted feature
extraction, model training, saved models, and single-video prediction all live
here.

Run commands from the repository root:

```powershell
python random_forest\extract_features.py --input-csv random_forest\metadata\test.csv --output-csv random_forest\features\test_features.csv
python random_forest\train.py
python random_forest\predict_video.py "C:\path\to\video.mp4"
```

`train.py` expects `train_features.csv`, `validation_features.csv`, and
`test_features.csv` in `random_forest/features/`. It writes the fitted scaler
and `random_forest.pkl` to `random_forest/models/`, plus evaluation results to
`random_forest/results/`.
