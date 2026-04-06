# Deep Learning for Tabular Data: A Comparative Study

## Project Overview
This repository contains a comprehensive benchmark study comparing specialized Deep Learning (DL) architectures for tabular data—specifically **TabNet** and **FT-Transformer**—against classical Machine Learning (ML) baselines and state-of-the-art Gradient Boosted Decision Trees (GBDTs). We evaluate these models across four distinct datasets representing various real-world challenges: regression, binary classification, large-scale multi-class classification, and highly imbalanced data.

## Contributors
* **CHEN SHU** (A0297139N)
* **WANG HAOYANG** (A0333322U)
* **MOU ZHENGYANG** (A0330675A)
* **LU LITIAN** (A0333227L)

---

## Experimental Methodology
To ensure rigorous comparison and statistical significance, we implemented the following protocol:
* **Data Splitting**: A consistent stratified split of $60\%$ Training, $20\%$ Validation, and $20\%$ Testing was applied across all datasets.
* **Stability**: Models were evaluated across 3 fixed random seeds to report mean performance and standard deviation.
* **Optimization**: Hyperparameters were tuned using **Optuna** to ensure each model reached its competitive potential.
* **Hardware**: Deep learning and GBDT models were accelerated using NVIDIA GPUs (e.g., Tesla T4 or RTX 5090).

---

## Key Experiments and Results

### 1. California Housing (Regression)
* **Task**: Predicting median house prices for 20,640 samples based on 8 numerical features.
* **Finding**: While FT-Transformer and ResNet achieved competitive $RMSE$, they did not significantly outperform **CatBoost** and **XGBoost**.
* **Efficiency**: GBDTs were substantially faster in training, whereas DL models required more extensive preprocessing and tuning.

### 2. Adult Income (Binary Classification)
* **Task**: Predicting whether annual income exceeds \$50K using 14 features (6 numerical, 8 categorical).
* **Finding**: **LightGBM** achieved the highest AUC ($0.9225$) and F1-score ($0.7012$).
* **Observation**: Training for FT-Transformer was approximately 30 times slower than LightGBM due to the computational overhead of self-attention.

### 3. Covertype (Multi-class Classification)
* **Task**: 7-class forest cover type prediction with 581,000 samples and 54 features.
* **Finding**: **LightGBM** was the most practical model, offering the best balance of efficiency and accuracy ($0.931$).
* **DL Challenges**: TabNet and FT-Transformer were highly sensitive to hyperparameters and significantly more computationally expensive.

### 4. Porto Seguro Safe Driver (Imbalanced Classification)
* **Task**: Predicting insurance claims in a dataset with only $3.6\%$ positive samples.
* **Finding**: **CatBoost** maintained the best predictive power with a mean AUC-ROC of $0.6367$.
* **Observation**: TabNet showed high variance across seeds (Std=0.0111), indicating instability in highly imbalanced tabular tasks.

---

## Technical Comparison Summary

| Metric | GBDTs (LightGBM/CatBoost) | Deep Learning (TabNet/FT-Transformer) |
| :--- | :--- | :--- |
| **Preprocessing** | Minimal (handles raw features/labels)  | High (requires scaling, embedding, imputation)  |
| **Training Speed** | Fast (50s for large data)  | Slow (>900s or 30x slower)  |
| **Stability** | High (Robust to seeds/tuning)  | Moderate to Low (Sensitive to init)  |
| **Interpretability**| High (Feature Importance)  | Low to Moderate (Black-box/Attention)  |

---

## Conclusion
Our research supports the verdict that for typical tabular datasets, **Gradient Boosted Decision Trees** remain the superior choice. While specialized Deep Learning models like FT-Transformer show potential, they currently require significantly more resources, complex preprocessing, and tuning without providing a consistent performance gain over optimized GBDTs.
