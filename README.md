# End-to-End Machine Learning Project: California Housing Price Prediction

## Project Overview

This project is a **complete, end-to-end machine learning workflow** built from scratch, demonstrating the practical application of industry best practices for building robust and reproducible ML systems. It focuses on the classic California Housing dataset to predict median house prices.

The workflow showcases proficiency in the entire ML lifecycle, including:
*   Building reproducible data pipelines.
*   Performing in-depth Exploratory Data Analysis (EDA).
*   Engineering meaningful features.
*   Training, evaluating, and fine-tuning multiple ML models.
*   Advanced use of **Scikit-Learn pipelines** and **custom transformers**.
*   Applying rigorous hyperparameter tuning using `GridSearchCV`.
*   Considering model persistence and deployment strategies.
*   Understanding real-world MLOps concepts (monitoring, versioning, drift handling).

This foundational project validates not only theoretical understanding but also the ability to implement practical, production-oriented ML solutions.

## 🎯 Objective

The primary goal is to **predict the median housing prices** in California districts using the publicly available California Housing dataset. The project meticulously walks through the entire lifecycle of an ML system—from raw data ingestion to a final, production-ready model artifact.

## 🔍 Key Implementation Details

The project follows a structured, multi-step approach to ensure a high-quality and maintainable solution:

### 1. Data Ingestion & Test Set Creation
*   **Automated Download:** The dataset is ingested programmatically via Python.
*   **Stratified Sampling:** A stratified train/test split is created based on income categories to ensure the test set is representative of the full data distribution, a critical step for reliable evaluation.
*   **Reproducibility:** Scikit-Learn utilities are used to ensure the split is reproducible.

### 2. Exploratory Data Analysis (EDA)
*   **Correlation Analysis:** Detected correlations using scatter plots and correlation matrices.
*   **Geospatial Visualization:** Visualized the geographical distribution of median house values (latitude & longitude) to identify spatial patterns.
*   **Key Predictors Identified:** Important features for prediction were identified, including:
    *   Median Income
    *   Proximity to the coast
    *   Rooms per household

### 3. Data Cleaning & Feature Engineering
A comprehensive, robust preprocessing pipeline was implemented using Scikit-Learn's `ColumnTransformer` and `Pipeline`:
*   **Missing Value Handling:** Imputation strategies were applied to handle missing data.
*   **Custom Feature Creation:** Engineered new, meaningful features such as `rooms_per_household` and `bedrooms_ratio`.
*   **Numerical Scaling:** Numerical features were scaled to prevent model bias.
*   **Categorical Encoding:** One-hot encoding was used for categorical attributes.
*   **Advanced Feature:** A **Custom `ClusterSimilarity` Transformer** was developed to compute cluster-based Radial Basis Function (RBF) similarity features, demonstrating advanced pipeline customization.

### 4. Model Training & Evaluation
Multiple models were trained to establish a performance baseline:
*   Linear Regression
*   Decision Tree
*   Random Forest
*   XGBoost (Included to validate modern ensemble methods)

All models were evaluated using **Root Mean Squared Error (RMSE)** and validated through **cross-validation** for reliable performance estimates.

### 5. Hyperparameter Tuning (`GridSearchCV`)
Rigorous hyperparameter tuning was performed using `GridSearchCV` to optimize the final model:
*   **Model Optimization:** Optimized `RandomForest` hyperparameters (`max_features`, `n_estimators`).
*   **Pipeline Optimization:** Crucially, the search included the hyperparameters of the **custom `ClusterSimilarity` transformer** within the preprocessing pipeline.
*   **MLOps Practice:** This step demonstrates an understanding of:
    *   Pipeline hyperparameter naming (e.g., `preprocessing__geo__n_clusters`).
    *   Efficient experiment management.
    *   **Avoiding data leakage** by encapsulating all steps within the pipeline.

### 6. Final Model & Evaluation
*   The best performing model was selected after fine-tuning.
*   Final evaluation was performed on the completely **held-out test set**.
*   The error distribution was examined, and predictions were compared against actual values.

### 7. Model Persistence (Production Preparation)
The final, trained model (including the full preprocessing pipeline) was saved using `joblib`:
```python
joblib.dump(final_model, "my_california_housing_model.pkl")
```
This section also documents the necessary steps for production readiness, including:
*   How to reload the model and handle custom transformers.
*   Considerations for building a REST API wrapper (e.g., using FastAPI or Flask).
*   Strategies for deployment to cloud platforms (e.g., Vertex AI, AWS Sagemaker).

## 🧠 Skills Demonstrated

This project serves as a portfolio piece, highlighting the following core competencies:

| Category | Skills Demonstrated |
| :--- | :--- |
| **ML System Design** | Building real, scalable ML systems, not just isolated notebook experiments. |
| **Advanced Scikit-Learn** | Expert use of `Pipeline`, `ColumnTransformer`, custom `BaseEstimator` transformers, and complex hyperparameter grids. |
| **MLOps Fundamentals** | Understanding of model versioning, monitoring model drift, test-set preservation, and automated retraining. |
| **Software Engineering** | Applying solid software engineering practices for modular, clean, and reproducible code. |
| **Communication** | Ability to communicate complex ML results clearly and professionally. |

## 🛠 Technologies Used

| Category | Tool/Library | Purpose |
| :--- | :--- | :--- |
| **Core Language** | Python | Primary programming language. |
| **Data Manipulation** | NumPy, Pandas | Efficient data structures and analysis. |
| **Machine Learning** | Scikit-Learn, XGBoost | Core ML framework and advanced ensemble modeling. |
| **Visualization** | Matplotlib, Seaborn | Exploratory Data Analysis and result presentation. |
| **Persistence** | Joblib | Saving and loading the final model artifact. |
| **Development** | Jupyter Notebook | Interactive development and documentation. |

## 📁 Project Structure

```
📦 End-to-End-ML-Project
├── data/                    # Raw and processed dataset
├── notebook.ipynb           # Full development notebook (contains all code)
├── models/                  # Saved model artifacts (joblib)
├── README.md                # Project documentation (this file)
└── images/                  # Visualizations for README (e.g., geospatial plots)
```

## 📜 License

This project is licensed under the **MIT License**.

