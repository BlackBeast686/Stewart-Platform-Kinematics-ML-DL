# Stewart Platform Kinematics — ML

Learning the **forward kinematics of a 6-DOF Stewart platform** with machine learning: given the six actuator leg lengths, predict the platform's pose — translation (X, Y, Z) and orientation (Roll, Pitch, Yaw). This repository compares classical ML regressors against graph-attention and hybrid models on the same data.

## Task & data

- **Inputs:** six leg lengths (`Leg_1` … `Leg_6`) from `Stw_PToP_Leg.dat`
- **Outputs:** six pose values (`X`, `Y`, `Z`, `Roll`, `Pitch`, `Yaw`) from `Line_6s_Th.dat`
- Features and targets are standardized, then split 70/30 into train/test.

Both `.dat` files (whitespace-delimited trajectory data) are included in this repository.

> **Note on scope:** These are research/experimental notebooks for comparing regression approaches, not a validated kinematics solver.

## Models

| Notebook | Model | Family |
|----------|-------|--------|
| `KNN, SVM, DecisionTree, RandomForest, GradientBoostingRegressor.ipynb` | KNN, SVM, Decision Tree, Random Forest, Gradient Boosting | Classical ML |
| `GAT_model_for_Stewart_Platform.ipynb` | Graph Attention Network | Graph NN |
| `Hybrid_GAT+REG.ipynb` | GAT + regression head | Hybrid |

The classical-ML notebook also stress-tests the regressors by injecting Gaussian noise and missing values (mean-imputed) before training. The GAT models treat samples as nodes in a fully connected graph and are trained with a combined MSE + MAE loss.

## Results

Overall test-set metrics as reported in each notebook:

| Model | R² | MAE | MSE |
|-------|-----|-----|-----|
| Hybrid GAT + Regression | 0.921 | 0.221 | 0.085 |
| GAT | 0.879 | 0.275 | 0.130 |
| KNN | 0.619 | 0.523 | 0.412 |
| Random Forest | 0.566 | 0.535 | 0.468 |
| SVM | 0.521 | 0.558 | 0.513 |
| Gradient Boosting | 0.506 | 0.534 | 0.528 |
| Decision Tree | 0.412 | 0.595 | 0.627 |

> The classical-ML figures are on the deliberately noised/missing-value version of the data, so they aren't directly comparable to the clean-data GAT runs; read them as two separate experiments. The hybrid GAT+regression model performs best overall. Figures are single runs — treat as indicative.

## Requirements

Developed in Google Colab. Main dependencies:

- `torch`
- `torch_geometric`
- `scikit-learn`
- `pandas`
- `numpy`

Install everything with:

```bash
pip install -r requirements.txt
```

## Getting started

1. Open any notebook in Google Colab or Jupyter.
2. Make sure `Stw_PToP_Leg.dat` and `Line_6s_Th.dat` are reachable at the path used in the notebook (`/content/` on Colab) — both files are in this repo.
3. Run the cells top to bottom. The GAT notebooks benefit from a GPU runtime.

## Repository structure

```
.
├── KNN, SVM, DecisionTree, RandomForest, GradientBoostingRegressor.ipynb
├── GAT_model_for_Stewart_Platform.ipynb
├── Hybrid_GAT+REG.ipynb
├── Stw_PToP_Leg.dat      # leg-length inputs (Dataset Used)
├── Line_6s_Th.dat        # pose outputs      (Dataset Used)
└── README.md
```

## License

Add a license of your choice (e.g. MIT) if you want others to reuse this work.
