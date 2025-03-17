

# Customer Churn Prediction using Random Forest Classifier

This project utilizes a Random Forest Classifier to predict customer churn. It demonstrates the entire machine learning pipeline, from data preprocessing to training, evaluating, and interpreting the model.

## Table of Contents
1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [Model and Methodology](#model-and-methodology)  
4. [Evaluation Metrics](#evaluation-metrics)  
5. [Results](#results)  
6. [Next Steps](#next-steps)  
7. [How to Run](#how-to-run)  

---

## Overview
Customer churn prediction is critical for businesses to identify and retain valuable customers. This project uses the **Random Forest Classifier** from the **scikit-learn** library to build a robust model that predicts churn with high accuracy.

---

## Dataset
The dataset contains various features related to customer behavior and demographics. Key steps in data preparation include:  
- Handling missing values (if any).  
- Dropping unnecessary columns (e.g., 'id').  
- Feature engineering: Creating features like `cons_price_ratio`, `gas_price_ratio`, `gas_cons_ratio`, and time-based features.  

---

## Model and Methodology
- **Model Used:** Random Forest Classifier from the **scikit-learn** package.  
- **Feature Engineering:** Polynomial and time-based features were engineered for better model performance.  
- **Training Process:**
  - Data was split into training and testing sets.  
  - Hyperparameters were tuned to optimize performance.
  - The model was trained using default parameters and later refined.

---

## Evaluation Metrics
The following metrics were used to evaluate model performance:  
- **Accuracy:** Overall correctness of predictions.  
- **Precision:** Relevant predictions for the positive class.  
- **Recall:** Ability to identify true positives (important for imbalanced datasets).  
- **F1-Score:** Harmonic mean of precision and recall.  
- **AUC-ROC:** Assesses the model’s ability to differentiate churners from non-churners.  

These metrics were chosen for their ability to provide a comprehensive view of the model's performance, particularly for imbalanced datasets like churn prediction.

---

## Results
The trained Random Forest Classifier achieved exceptional performance:  
- Accuracy: **0.999857**  
- Precision: **1.0**  
- Recall: **0.998529**  
- F1-Score: **0.999264**  
- AUC-ROC: **1.0**  

While the results indicate excellent performance, the high scores might suggest **overfitting**, which needs further exploration.

---

## Next Steps
1. Investigate potential overfitting using **cross-validation** and by reducing model complexity.  
2. Perform **hyperparameter tuning** to balance model accuracy and generalization.  
3. Evaluate the model on a separate, held-out test set for robust validation.  

---

## How to Run
1. Clone this repository:  
   ```bash
   git clone <repository-link>
   cd <repository-name>
   ```  
2. Ensure you have the required dependencies installed:
   - Python  
   - scikit-learn  
   - pandas  
   - numpy  

   Install them with:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Run the notebook:
   - Open the Jupyter Notebook provided in the repository.  
   - Follow the sequential cells to preprocess data, train the model, and evaluate its performance.  

