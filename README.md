<img width="6000" height="3000" alt="etienne-martin-2_K82gx9Uk8-unsplash" src="https://github.com/user-attachments/assets/ec5193ac-5c68-40c4-aa64-2d6f6f1a9689" />

![Python](https://img.shields.io/badge/Python-%2B-blue?style=flat-square&logo=python)
![Static Badge](https://img.shields.io/badge/Jupyter-Notebook-orange)
![NumPy](https://img.shields.io/badge/numpy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3399FF?style=flat-square&logo=seaborn&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-0088FF?style=flat-square&logo=xgboost&logoColor=white)


# Churn Prediction of a Bank 
This project analyzes customer churn in a banking context to develop a Machine Learning model that can successfully identify clients at risk of leaving.. By identifying customers at high risk of leaving, the bank can implement targeted retention strategies, thereby reducing customer attrition and safeguarding revenue.

## Table of Contents
- [Business Problem & Objectives](#business-problem-&objectives)
- [Dataset](#dataset)
- [Exploratory Data Analysis(EDA)](#exploratory-data-analysis-(eda))
- [Methodology](#methodology)
- [Final Model](#final-model)
    -   [Final Perfomance](#final-perfomance)
    -   [Most Important Features](#final-perfomance)
-   [Business Insights](#business-insights)
-   [Recommendations](#recommendations)  


## Business Problem & Objectives
Customer churn is a critical challenge for banks, directly impacting revenue and profitability growth. In the banking context, acquiring clients is significantly more costly than retaining customers. Retaining customers ensures predictable revenue streams, fosters brand loyalty, and considerably boosts the bank's long-term profitability. This project aims to address this by:
1. **Identifying customer profiles** at higher risk of churn.
2. **Identify which features are the strongest** indicators of churning.
3. **Developing a robust machine learning model** to predict which customers will churn.

The ultimate goal is to provide actionable insights and a predictive tool that can assist the bank in proactively intervening and reducing churn.

## Dataset 
The dataset contains information on 10,000 bank clients, with churn as the target variable (1 = Churn, 0 = No Churn).

Features:

The dataset includes the following features:

- **customer_id**: customer identifier
- **credit_score**: customer credit score
- **country**: country of residence
- **gender**: customer gender
- **age**: customer age
- **tenure**: number of years the customer has been with the bank
- **balance**: account balance
- **products_number**: number of bank products used
- **credit_card**: whether the customer has a credit card
- **active_member**: whether the customer is an active member
- **estimated_salary**: estimated salary
- **churn**: target variable  
  - `1` = churn  
  - `0` = no churn

The dataset exhibits a churn rate of approximately 20.37%, indicating a moderately imbalanced dataset, which was considered during model selection and evaluation. 

**Source:** This dataset was originally obtained from [Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data).

## Exploratory Data Analysis (EDA)

The exploratory analysis revealed several important patterns related to churn:

## Key Findings
### By Country

**Country**: **Germany** consistently shows the highest churn rate, approximately **32%**, which is nearly double that of France (16%) and Spain (17%). This suggests unique regional challenges or market dynamics.

  <img width="2052" height="1153" alt="churn_country" src="https://github.com/user-attachments/assets/5a1fdacd-9713-4995-87ae-f66859b20ade" />

  *Figure 1: Churn Rate distribution across different countries, highlighting Germany's elevated rate.*

### By Age

**Age**: Age is a significant predictor of churn. The churn rate **peaks in the 50-60 age range**, where more than half of the customers in this group are likely to leave the bank. Conversely, younger clients (under 30) show the lowest churn.

   <img width="4153" height="1452" alt="churn_age" src="https://github.com/user-attachments/assets/bb75758b-30a3-4c84-90a0-46bde0ae40d7" />

  *Figure 2: Churn Rate distribution across various age groups, showing a peak in middle-aged customers.*

### By Products

  **Number of Products**: This feature exhibits a **non-linear relationship** with churn:
  - Customers with **2 products** have the **lowest churn rate** (around 7.6%).
  - Customers with **1 product** or **3+ products** show significantly **higher churn rates** (with 3-4 products reaching 82-100% churn). This may indicate dissatisfaction or forced acquisition of unwanted products.

   <img width="2352" height="1154" alt="churn_pro" src="https://github.com/user-attachments/assets/374bffb6-b645-468e-abe6-cc8679ad47ca" />
   
   *Figure 3: Churn Rate based on the number of bank products held by customers, illustrating the non-linearity.*
  
  <img width="2353" height="1455" alt="churn_num" src="https://github.com/user-attachments/assets/0d90fa16-7564-4ca8-b0ac-5242cd4ec2a6" />
     
   *Figure 4: Churn Rate based on the number of bank products held by customers, excluding sample pool < 500 clients.*

### By Gender
**Gender**: **Women** were shown to have a higher churn rate than men, at around **25%**

<img width="1752" height="1451" alt="churn_gender" src="https://github.com/user-attachments/assets/e20d339e-5326-437b-bb30-a355ce4bf2d0" />

*Figure 5: Churn Rate based on gender of the customers.*

### By Active Membership

**Active Members**: **Active customers are consistently less likely to churn** compared to inactive members. This suggests that engagement is a strong indicator of customer loyalty.

### By Balance Behavior

**Balance**: Customers with **positive balances** tend to show a higher churn rate than those with zero balance. This could imply that clients with more assets have more options and are more willing to explore other banking services.

### By Credit Score

**Credit Score**: No significant trend or strong predictive power was observed for credit score, with churn rates remaining relatively consistent across different score ranges.

### Highest-Risk Customer Profile

The highest-risk churn profile identified in the analysis was:

- **Female**
- **Based in Germany**
- **Middle-aged**
- **Inactive member**
- **Positive balance**
- **1 product or 3+ products**

##  Methodology

### Preprocessing

Variables were grouped according to their proper type for appropriate handling:

-   **Numerical features**: Median imputation + Standardization (**StandardScaler**)
-   **Categorical features**: Mode imputation + One-Hot Encoding (**OneHotEncoder**)
-   **Binary features**: Passed through without transformation, as they are already in 1/0 values

A **Pipeline** with **ColumnTransformer** was used to prevent data leakage by ensuring transformations were learned only on the training set and then applied correctly to the test set.

### Modeling

Three classification models were tested and compared:

-   **Logistic Regression** (Linear model)
-   **Random Forest** (Non-linear ensemble)
-   **XGBoost** (Gradient boosting)

Hyperparameter tuning was performed using **RandomizedSearchCV** with **StratifiedKFold** to handle the imbalanced dataset. Class weights were adjusted (**class_weight="balanced"** for LR/RF, **scale_pos_weight** for XGBoost) to prioritize the minority class (churn).

### Model Evaluation Curves

<img width="4152" height="1451" alt="churn_modelcomparison" src="https://github.com/user-attachments/assets/2d614dfe-200e-42d4-93c4-bed683be2412" />

*Figure 6: Comparing the ROC Curve and Precision-Recall in the three classification models tested* 

## Final Model

The final model selected for the project was **XGBoost**, due to its strong overall performance in identifying churners.

### Final Performance

- **AUC-ROC:** 0.8665
- **AUC-PR:** 0.7159
- **Recall (Churn):** 0.806
- **Precision (Churn):** 0.462
- **Decision threshold:** 0.45

The threshold was adjusted to prioritize recall, as in churn prediction it is usually more important to identify as many at-risk customers as possible

#### Confusion Matrix

<img width="1707" height="1154" alt="churn_finalmatrix" src="https://github.com/user-attachments/assets/cf4aa8ef-dfaf-4c40-821f-f1af57d6349b" />

*Figure 7: Confusion Matrix for the selected XGBoost model with a threshold of 0.45.*

### Most Important Features

According to the XGBoost feature importance analysis, the most relevant variables were:

<img width="2952" height="1451" alt="churn_features" src="https://github.com/user-attachments/assets/edc23921-2ab3-462e-9bc7-5b9f3999064e" />

*Figure 8: Top 10 most important features for the **XGBoost** model*

1. **products_number**
2. **age**
3. **active_member**
4. **country**  
   - especially customers from **Germany**

## Business Insights

The analysis suggests that churn is influenced more by **behavioral and profile-related factors** than by credit score alone.

### Main business implications

- Customers with low engagement should be targeted with retention strategies
- Germany deserves special attention due to its higher churn rate
- Product diversification may help reduce churn
- Older customers should be monitored more closely
- Customers with positive balances may be more likely to compare offers and switch banks

## Recommendations

- Create targeted retention campaigns for high-risk customer segments
- Investigate the causes of high churn in Germany
- Improve engagement among inactive customers
- Review product strategy for customers with only one product or with multiple products
- Use the model to prioritize outreach to customers most likely to leave

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

## Repository Structure
- `bank_customer_churn_prediction.csv` — csv file with the dataset
- `Projeto_Banco_Churn.ipynb` — main notebook with EDA, preprocessing, modeling, and evaluation
- `README.md` — project documentation

## Future Improvements

- Deploy the model as an API
- Add model explainability with SHAP
- Perform A/B testing to measure business impact
- Create a dashboard to monitor churn risk
