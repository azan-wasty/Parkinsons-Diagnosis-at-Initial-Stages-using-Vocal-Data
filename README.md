# Parkinsons Disease Voice Classification

End-to-end notebook pipeline for classifying Parkinsons disease from sustained voice measurements (UCI Parkinsons dataset). The notebook performs cleaning, feature engineering, exploratory analysis, model training with aggressive regularization, and saves artifacts for reuse.

## Data
- Expected file: `parkinsons.data` at the project root (CSV from the UCI Parkinsons Disease dataset).
- Target column: `status` (0 = healthy, 1 = Parkinsons).
- Identifier column `name` (if present) is dropped during cleaning.

## Environment
```
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy seaborn matplotlib scikit-learn psutil
```

## How to Run
1) Place `parkinsons.data` in the repository root.
2) Open `Parkinsons.ipynb` in VS Code and run cells sequentially. All outputs are written to `parkinsons_outputs/` (created automatically).
	- If you prefer running as a script, export the notebook to a `.py` file and execute it: `python Parkinsons.py`.

## Pipeline (notebook cells)
- Load data: read CSV, report shape, dtypes, preview.
- Feature engineering: derived jitter/shimmer aggregates, frequency range/CV, HNR-to-NHR ratio, perturbation and interaction features.
- EDA: target distribution, PPE scatterplot, key feature histograms/boxplots, correlation heatmap (saved under `parkinsons_outputs/eda_plots/`).
- Cleaning: drop `name`, remove duplicates, fill numeric nulls with median, correlation-based top-15 feature selection.
- Target validation: ensure `status` exists with two classes.
- Scaling: StandardScaler fitted on features (target excluded).
- Split: stratified 80/20 train-test split.
- Hyperparameter tuning: GridSearchCV for RandomForestClassifier and LogisticRegression with strong regularization to limit overfitting.
- Evaluation: train/test metrics, overfitting gap, F1, ROC-AUC (when available), confusion matrix; learning curves are saved.
- Resource usage: inference time and memory via psutil.
- Artifact saving: models (`Random_Forest.pkl`, `Logistic_Regression.pkl`, `best_model.pkl`), `scaler.pkl`, `feature_names.pkl`, feature importance plot for Random Forest.

## Outputs
Created under `parkinsons_outputs/`:
- `eda_plots/` (PNG plots: target distribution, PPE scatter, feature distributions, boxplots, correlation heatmap).
- `hyperparameter_tuning.csv`, `model_metrics.csv`, `resource_usage.csv`.
- `learning_curve_<Model>.png`, `feature_importance_Random_Forest.png`.
- Serialized artifacts: model pickles, scaler, feature names, best model.

## Notes
- Standardization is applied to features before model training; target is excluded from scaling.
- Default split is 80/20 with stratification; adjust `test_size` in `safe_train_test_split` if needed.
- Hyperparameter grids favor shallow trees and strong regularization to reduce overfitting; expand grids if you want a higher-variance search.
