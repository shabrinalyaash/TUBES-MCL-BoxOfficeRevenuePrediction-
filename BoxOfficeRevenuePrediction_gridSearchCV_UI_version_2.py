import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, accuracy_score

# Set page config
st.set_page_config(page_title="Box Office Revenue Prediction", layout="wide")

# Title
st.markdown("""
    <h1 style='text-align: center;'>Box Office Revenue Prediction</h1>
""", unsafe_allow_html=True)

# Upload file
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')

    # Dataset Preview
    st.markdown("""
        <h2 style='text-align: center;'>Dataset Preview</h2>
    """, unsafe_allow_html=True)
    st.write(df.head())

    # Short description
    st.info("""
    This dataset includes information about movie releases such as number of theaters, release days, MPAA ratings, genres, and domestic revenue. 
    The goal is to predict box office performance based on these features.
    """)

    # Preprocessing
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

    features = df.drop(['title', 'domestic_revenue'], axis=1)
    target = df['domestic_revenue'].values

    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1, random_state=22)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Model Training and Evaluation
    st.markdown("""
        <h2 style='text-align: center;'>Model Training and Evaluation</h2>
    """, unsafe_allow_html=True)

    # Train Models
    param_grid_lr = {'fit_intercept': [True, False]}
    grid_lr = GridSearchCV(LinearRegression(), param_grid_lr, cv=5, scoring='neg_mean_squared_error')
    grid_lr.fit(X_train, Y_train)
    best_lr = grid_lr.best_estimator_
    lr_val_preds = best_lr.predict(X_val)
    lr_val_error = mean_squared_error(Y_val, lr_val_preds)

    binary_target = (Y_train > Y_train.mean()).astype(int)
    binary_val_target = (Y_val > Y_val.mean()).astype(int)
    param_grid_logistic = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    grid_logistic = GridSearchCV(LogisticRegression(), param_grid_logistic, cv=5, scoring='accuracy')
    grid_logistic.fit(X_train, binary_target)
    best_logistic = grid_logistic.best_estimator_
    logistic_val_preds = best_logistic.predict(X_val)
    logistic_val_accuracy = accuracy_score(binary_val_target, logistic_val_preds)

    param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
    grid_svm = GridSearchCV(SVR(), param_grid_svm, cv=5, scoring='neg_mean_squared_error')
    grid_svm.fit(X_train, Y_train)
    best_svm = grid_svm.best_estimator_
    svm_val_preds = best_svm.predict(X_val)
    svm_val_error = mean_squared_error(Y_val, svm_val_preds)

    # Layout 3 columns for visualizations
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<h4 style='text-align: center;'>Linear Regression</h4>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.scatter(Y_val, lr_val_preds)
        ax1.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], 'r--')
        ax1.set_xlabel('Actual Revenue')
        ax1.set_ylabel('Predicted Revenue')
        ax1.set_title('Linear Regression Predictions')
        st.pyplot(fig1)

    with col2:
        st.markdown("<h4 style='text-align: center;'>Logistic Regression</h4>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.hist(binary_val_target - logistic_val_preds, bins=5)
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Count')
        ax2.set_title('Logistic Regression Prediction Errors')
        st.pyplot(fig2)

    with col3:
        st.markdown("<h4 style='text-align: center;'>SVM</h4>", unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.scatter(Y_val, svm_val_preds)
        ax3.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], 'g--')
        ax3.set_xlabel('Actual Revenue')
        ax3.set_ylabel('Predicted Revenue')
        ax3.set_title('SVM Predictions')
        st.pyplot(fig3)

    # Summary
    st.markdown("""
        <h2 style='text-align: center;'>Summary of Model Performance</h2>
    """, unsafe_allow_html=True)

    st.success(f"Linear Regression Validation Error (MSE): {lr_val_error:.2f}")
    st.success(f"Logistic Regression Validation Accuracy: {logistic_val_accuracy:.2f}")
    st.success(f"SVM Validation Error (MSE): {svm_val_error:.2f}")
