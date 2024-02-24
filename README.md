# Instacart Store - Predict If Customer will reorder a Product - Group Project
[Streamlit App](https://predicting-appucts-reordered-by-customer-7852zyrzsf2axkrqwx8l8.streamlit.app/)

# Description 
**Objective:** Predict whether a product will be reordered by a Instacart customer (Supervised Classification Problem - Target Variable (Reordered))

**Significance:**

**Tailored Customer Experience:** Personalize the app interface and recommendations to resonate with individual shopping habits and preferences.

**Enhanced Inventory Management:** Use predictions to inform stock levels, reducing waste and ensuring availability of frequently reordered items.

**Operational Efficiency:** Implement intelligent route planning for deliveries based on predicted orders, saving time and reducing logistical costs

**Hypothesis:** User's order history can signal their product preferences and future buys

# Data Source and Description 

[Kaggle Link](https://www.kaggle.com/c/instacart-market-basket-analysis/overview)

This dataset contains information about 3 million grocery orders from 200,000 users. Each user in the dataset has a history of 4 to 100 orders. The order information is extensive, including timestamps, the sequence of products ordered, whether items are reordered, and details about the customers themselves. Additionally, the dataset includes product information, providing product names along with the aisles and departments they belong to. For ease of reference, aisles and departments are identified by unique IDs

# Modelling Approach

* Preprocessing & EDA

* Feature Engineering

* Distribution Shift & Imbalanced Data

* Feature Selection & Dimension Reduction

* Modeling & Evaluation

# Data Preprocessing and EDA

* Dimension tables are merged to fact table using primary keys.

* Merged Dataset is analyzed for missing values, duplicates, outliers.

* [Full report](https://github.com/TashfeenAhmed12/Predicting-Products-Reordered-by-Customer/blob/28cfcfbe4a25c53c7d99834c899af0ef77eee74e/data_profiling_report.html)


# Feature Engineering/Feature Selection

Following metrics were created to capture more details for each user

* Total orders
* Average number of products per order
* Average days between orders
* Most common order day of week
* Most Common Order Hour

Handling Categorical Features:

* One Hot Encoding
* Ordinal and Frequency Encoding

Moreover, inbalanced dataset was handled and information leakage analysis was conducted

Feature Selected was conducted using Lasso and Random Forest

# Results
Following models were trained and compared to find the best model , models compared - XGB Classifier,LGBM Classifier, CatBoost, Stacking Classifier

Final model selected was CatBoost and it was trained using Optuna with Train-Validation-Test Approach acheiving high accuracy and f1 score

![image](https://github.com/TashfeenAhmed12/Predicting-Products-Reordered-by-Customer/assets/76031323/8362e746-c550-413c-961c-63a7b98db420)


**Threats to Validity:** 

* Change in environmental factors or Data Integrity may affect results. For instance, economic shifts, new market entrants, or changes in consumer preferences could alter purchasing patterns

* If the underlying data generating process changes over time model drift might happen

**Next Steps**
  
* Optuna to optimize multiple scores concurrently

* To further optimize the models use custom loss function such as focal loss

* Integrate advance feature engineering to capture aggregated measures
