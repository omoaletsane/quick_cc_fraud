# Credit Card Fraud Detection (Quick Model)

## Overview
This project implements a machine learning pipeline for credit card fraud detection using the well-known Credit Card Fraud Dataset (Pozzolo et al., 2015).  
The primary aim is to design a computationally efficient model suitable for environments with limited resources while still demonstrating the methodological steps needed for fraud detection research.

The project applies the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology, focusing on:
1. Business Understanding  
2. Data Understanding  
3. Data Preparation  
4. Modeling  
5. Evaluation  

Deployment was excluded given the experimental scope.

---

## Dataset
* **Source**: ULB Machine Learning Group (n.d.).  
* **Transactions**: 284,807 (Europe, 2013)  
* **Frauds**: 492 (~0.17%)  
* **Features**:  
  - 28 PCA-transformed components (V1–V28)  
  - Time and Amount (scaled before modeling)  
  - Class (0 = legitimate, 1 = fraud)  

**Downsampling applied due to imbalance and computational limits:**
- All 492 fraud cases  
- 15,000 randomly sampled non-fraud cases  

---

## Methodology

### Preprocessing
- **Scaling**: StandardScaler for Time and Amount.  
- **Balancing**: Downsampling of majority class.  
- **Feature Handling**: PCA features retained without modification.  

### Models Implemented
1. **Logistic Regression (LR)**  
   - Linear baseline model  
   - Hyperparameter tuned (`C` regularization strength)  

2. **Decision Tree (DT)**  
   - Non-linear, interpretable classifier  
   - Hyperparameters tuned (`max_depth`, `min_samples_split`)  

---

## Optimization
- Hyperparameter tuning with **GridSearchCV** (3-fold cross-validation).  
- Metrics prioritized: **Precision, Recall, F1-score** for the fraud class.  

---

## Results

| Model              | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | Accuracy |
|--------------------|-------------------|----------------|------------|----------|
| Logistic Regression | 0.55              | 0.89           | 0.68       | 0.97     |
| Decision Tree       | 0.48              | 0.90           | 0.62       | 0.96     |

- Logistic Regression slightly outperformed Decision Tree in fraud precision (fewer false alarms).  
- Both models achieved high recall, showing effectiveness in capturing fraud cases.  
- Accuracy remains high due to the large number of non-fraudulent transactions.  

Outputs are saved in the **`outputs_fast/`** directory, including:
- Confusion matrices  
- Performance tables  
- Run metadata  

---

## Limitations
Originally, a full end-to-end model (`credit_card_fraud_end_to_end.py`) was attempted with stratified sampling, SMOTE, hyperparameter tuning (GridSearchCV, RandomizedSearchCV), and threshold optimization.  

However, due to limited computational resources, the experiment could not complete.  
This project therefore focuses on a lighter pipeline, which, while informative, does not capture the full potential of ensemble and boosting methods (e.g., Random Forest, XGBoost).  

---

## How to Run
```
1. Clone the repository:
```bash
git clone https://github.com/omoaletsane/quick_cc_fraud.git
cd quick_cc_fraud

2. Create a virtual environment:
```powershell
python -m venv .venv

2.1. Activate the environment:
```bash
.venv\Scripts\activate

3. Install dependencies:
```bash
pip install -r requirements.txt

4. Run the model:
python quick_cc_fraud.py
```
---

## References

* Hajek, P., Abedin, M. Z., & Sivarajah, U. (2022). Fraud detection in mobile payment systems using an XGBoost-based framework. Information Systems Frontiers, 1–19.

* Machine Learning Group – ULB (n.d.). Credit Card Fraud Detection. Dataset.

* Pozzolo, A.D., Caelen, O., Johnson, R.A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. IEEE SSCI, pp. 159–166.

* Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.

* Velarde, G., Weichert, M., Deshmunkh, A., Deshmane, S., Sudhir, A., Sharma, K., & Joshi, V. (2024). Tree boosting methods for balanced and imbalanced classification and their robustness over time in risk assessment. ISWA, 200354.
