🏡 End-to-End Machine Learning Project — California Housing Price Prediction

This project is a complete end-to-end machine learning workflow, built from scratch following industry best practices.
It demonstrates my ability to:

Build reproducible data pipelines

Perform exploratory data analysis (EDA)

Engineer meaningful features

Train, evaluate, and fine-tune ML models

Use Scikit-Learn pipelines and custom transformers

Apply hyperparameter tuning (GridSearchCV)

Handle model persistence and deployment considerations

Understand real-world ML operations (monitoring, versioning, drift handling)

This is a foundational ML project that shows not only that I understand the theory, but that I can implement practical, production-oriented ML systems.

📁 Project Structure
📦 End-to-End-ML-Project
├── data/                    # Raw and processed dataset
├── notebook.ipynb           # Full development notebook
├── models/                  # Saved model artifacts (joblib)
├── README.md                # Project documentation
└── images/                  # Visualizations for README

🎯 Objective

Predict median housing prices in California districts using the classic California Housing dataset.

This project walks through the entire lifecycle of an ML system—from raw data to a production-ready model.

🔍 Key Features & What I Implemented
1️⃣ Data Ingestion & Test Set Creation

Automated dataset download via Python.

Created a stratified train/test split based on income categories to ensure representative sampling.

Ensured reproducibility using Scikit-Learn utilities.

2️⃣ Exploratory Data Analysis (EDA)

Detected correlations using scatter plots and correlation matrices.

Visualized geospatial patterns in median house values (latitude & longitude).

Identified important predictors such as:

Median Income

Proximity to coast

Rooms per household

3️⃣ Data Cleaning & Feature Engineering

Implemented a full preprocessing pipeline:

✔ Handling missing values via imputation
✔ Custom feature creation (e.g., rooms_per_household, bedrooms_ratio)
✔ Scaling numerical features
✔ One-hot encoding for categorical attributes
✔ Custom ClusterSimilarity Transformer to compute cluster-based RBF similarity features
✔ Integrated everything using a Scikit-Learn ColumnTransformer & Pipeline

4️⃣ Model Training

Trained multiple models to compare baseline performance:

Linear Regression

Decision Tree

Random Forest

XGBoost (validates modern ML methods)

Evaluated using RMSE and validated through cross-validation for reliable performance estimates.

5️⃣ Hyperparameter Tuning (GridSearchCV)

Used GridSearchCV to optimize:

RandomForest hyperparameters (max_features, n_estimators)

ClusterSimilarity hyperparameters inside the preprocessing pipeline

Search executed through 3-fold CV with custom scoring (neg_root_mean_squared_error)

Demonstrates understanding of:

Pipeline hyperparameter naming (preprocessing__geo__n_clusters)

Efficient experiment management

Avoiding data leakage through encapsulated pipelines

6️⃣ Final Model & Evaluation

After fine-tuning:

Selected best model

Evaluated on held-out test set

Examined error distribution

Compared predictions vs actual values

7️⃣ Model Persistence (Production Preparation)

Saved final model using joblib:

joblib.dump(final_model, "my_california_housing_model.pkl")


Documented how to:

Reload model in production

Handle custom transformers on reload

Build REST API wrapper (FastAPI or Flask)

Deploy the model to cloud platforms (Vertex AI, AWS Sagemaker)

🧠 What This Project Demonstrates About My Skills

This project shows that I can:

✔ Build real, scalable ML systems — not just notebook experiments
✔ Use Scikit-Learn at an advanced level (pipelines, transformers, hyperparameter grids)
✔ Apply solid software engineering practices
✔ Understand MLOps fundamentals such as:

model versioning

monitoring model drift

test-set preservation

automated retraining

✔ Communicate ML results clearly and professionally
🛠 Technologies Used

Python

NumPy / Pandas

Matplotlib / Seaborn

Scikit-Learn

Joblib

XGBoost

Jupyter Notebook

🚀 Future Improvements

Deploy as an interactive Streamlit web app

Build a FastAPI prediction API

Add monitoring dashboards (data drift & performance decay)

Experiment with deep learning approaches for non-linear patterns

📜 License

MIT License
