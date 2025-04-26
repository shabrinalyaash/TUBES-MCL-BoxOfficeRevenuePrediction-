import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, accuracy_score

# Title of the app
st.title("Box Office Revenue Prediction")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    st.write("### Dataset Preview")
    st.write(df.head())

    # Preprocessing
    st.write("### Preprocessing Data")
    df.drop(['world_revenue', 'opening_revenue'], axis=1, inplace=True)
    df.drop('budget', axis=1, inplace=True)
    for col in ['MPAA', 'genres']:
        df[col] = df[col].fillna(df[col].mode()[0])
    df.dropna(inplace=True)
    df['domestic_revenue'] = df['domestic_revenue'].astype(str).str[1:].str.replace(',', '').astype(float)

    # Encode categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Features and target
    features = df.drop(['title', 'domestic_revenue'], axis=1)
    target = df['domestic_revenue'].values

    # Train-test split
    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1, random_state=22)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Model Training and Evaluation
    st.write("### Model Training and Evaluation")

    # Linear Regression
    st.write("#### Linear Regression")
    param_grid_lr = {'fit_intercept': [True, False]}
    grid_lr = GridSearchCV(LinearRegression(), param_grid_lr, cv=5, scoring='neg_mean_squared_error')
    grid_lr.fit(X_train, Y_train)
    best_lr = grid_lr.best_estimator_
    lr_val_error = mean_squared_error(Y_val, best_lr.predict(X_val))
    st.write(f"Validation Error (MSE): {lr_val_error}")

    # Logistic Regression
    st.write("#### Logistic Regression")
    binary_target = (Y_train > Y_train.mean()).astype(int)
    binary_val_target = (Y_val > Y_val.mean()).astype(int)
    param_grid_logistic = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    grid_logistic = GridSearchCV(LogisticRegression(), param_grid_logistic, cv=5, scoring='accuracy')
    grid_logistic.fit(X_train, binary_target)
    best_logistic = grid_logistic.best_estimator_
    logistic_val_accuracy = accuracy_score(binary_val_target, best_logistic.predict(X_val))
    st.write(f"Validation Accuracy: {logistic_val_accuracy}")

    # SVM
    st.write("#### Support Vector Machine (SVM)")
    param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
    grid_svm = GridSearchCV(SVR(), param_grid_svm, cv=5, scoring='neg_mean_squared_error')
    grid_svm.fit(X_train, Y_train)
    best_svm = grid_svm.best_estimator_
    svm_val_error = mean_squared_error(Y_val, best_svm.predict(X_val))
    st.write(f"Validation Error (MSE): {svm_val_error}")

    # Summary of Results
    st.write("### Summary of Model Performance")
    st.write(f"Linear Regression Validation Error (MSE): {lr_val_error}")
    st.write(f"Logistic Regression Validation Accuracy: {logistic_val_accuracy}")
    st.write(f"SVM Validation Error (MSE): {svm_val_error}")