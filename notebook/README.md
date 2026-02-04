# Telco Customer Churn Analysis
A comprehensive machine learning project analyzing customer churn patterns in a telecommunications company dataset. This project includes exploratory data analysis (EDA) and the development of multiple classification models to predict customer churn.

 ## Dataset Overview
- Total Customers: 7,043
- Features: 20 (after removing customerID)
- Target Variable: Churn (Yes/No)
- Churn Rate: 26.5% (1,869 churned vs 5,174 retained)

## Key Observations

### 1.Demographic Insights

- Gender: Churn is relatively balanced across genders, with no significant gender-based churn pattern
- Senior Citizens: Higher churn rate among senior citizens (~42%) compared to non-seniors (~24%)

### 2.Family Status:

- Customers with partners show lower churn rates
- Customers without dependents have significantly higher churn rates


### 3.Contract Type (Strongest Predictor):

- Month-to-month contracts: ~43% churn rate
- One-year contracts: ~11% churn rate
- Two-year contracts: ~3% churn rate
- Insight: Long-term contracts dramatically reduce churn


### 4.Internet Service:

- Fiber optic users show highest churn (~42%)
- DSL users have lower churn (~19%)
- Customers without internet service have lowest churn (~7%)


### 5.Add-on Services:

- Customers WITHOUT online security, backup, device protection, or tech support show significantly higher churn
- These value-added services act as retention factors



### 6.Financial Patterns

- Monthly Charges: Churned customers tend to have higher monthly charges (median ~$80 vs ~$65)
- Total Charges: Churned customers have lower total charges, indicating shorter tenure
- Tenure: Strong negative correlation with churn - newer customers are much more likely to leave

## Model Development
## Preprocessing Steps

- One-hot encoding for categorical variables
- Standard scaling for numerical features
- SMOTE for handling class imbalance
- PCA for dimensionality reduction (95% variance retained → 4,965 components)

## Models Evaluated
```
Model                   ROC-AUC   Precision(Churn)  Recall(Churn)  F1-Score(Churn) Overall Accuracy
Logistic Regression      0.812      0.62                0.45            0.52           0.78
SVM                      0.796      0.44                0.81            0.57           0.67
Naive Bayes              0.491      0.26                0.94            0.41           0.28
KNN                      0.537      0.69                0.05            0.09           0.74
Decision Tree            0.644      0.27                0.93            0.42           0.33
```
## Best Model: Logistic Regression
Logistic Regression emerges as the best model with:
  - Highest ROC-AUC Score: 0.812
  - Best Balance: Good precision-recall tradeoff
  - Highest Overall Accuracy: 78%
  - Interpretability: Easy to understand feature importance
  - Production Ready: Fast inference and stable predictions

## Performance Metrics
```
              precision    recall  f1-score   support
           0       0.82      0.90      0.86      1035
           1       0.62      0.45      0.52       374
    accuracy                           0.78      1409
```
## Why Logistic Regression?

- Balanced Performance: While SVM has higher recall, its lower precision means more false positives
- Business Context: 81% ROC-AUC provides reliable probability estimates for targeted retention campaigns
- Actionable Insights: Coefficient interpretation helps identify key churn drivers
- Computational Efficiency: Fast training and prediction for production deployment

## Business Recommendations

- Contract Incentives: Encourage month-to-month customers to switch to longer-term contracts
- Value-Added Services: Promote online security, backup, and tech support packages
- New Customer Focus: Implement enhanced onboarding and early engagement programs (first 0-12 months)
- Fiber Optic Investigation: Investigate quality/pricing issues with fiber optic service
- Senior Citizen Programs: Develop specialized retention programs for senior citizens
- Pricing Strategy: Review pricing for high monthly charge customers

## Technologies Used

- Python 3.13
- Data Analysis: pandas, numpy
- Visualization: matplotlib, seaborn
- Machine Learning: scikit-learn
- Imbalanced Data: imbalance-learn (SMOTE)

## Project Structure
```
├── data/
│   └── Telco_customer_churn.csv
├── notebook/
│   └── churn_analysis_EDA.ipynb
│   └── churn_analysis_EDA.ipynb
└── README.md
```
## Key Takeaways

- Contract length is the strongest predictor of churn
- Customer tenure inversely correlates with churn risk
- Add-on services significantly improve retention
- Logistic Regression provides the best balanced performance for churn prediction
- 73% of customers at risk of churn can be correctly identified
