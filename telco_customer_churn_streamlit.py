import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Title
st.title("Telco Customer Churn Prediction Dashboard")

# Sidebar - Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Data Overview", "Predict Churn", "Insights Dashboard"])

# Sidebar - Upload CSV data
st.sidebar.header("Upload your CSV data")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# Function to load data
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to preprocess data
def preprocess_data(df):
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
    original_data = df.copy()
    label_encoders = {}
    
    # Ensure Churn is properly encoded (0 for No, 1 for Yes)
    if 'Churn' in df.columns and df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        original_data['Churn'] = df['Churn']  # Sync the change with original_data

    # Encode other categorical variables
    for column in df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    
    return df, original_data, label_encoders

# Function to reverse encode categorical columns using the label encoders
def reverse_encode(df, label_encoders):
    for column, le in label_encoders.items():
        if column in df.columns:
            df[column] = le.inverse_transform(df[column])
    return df

# Load and preprocess the data if uploaded
if uploaded_file is not None:
    df = load_data(uploaded_file)
    df, original_data, label_encoders = preprocess_data(df)

    # Only show Feature Selection if "Predict Churn" is selected
    if option == "Predict Churn":
        # Sidebar - Feature Selection for Prediction (Remove Churn from features)
        features_without_churn = [col for col in df.columns if col != 'Churn']
        st.sidebar.header("Feature Selection")
        selected_features = st.sidebar.multiselect("Select features to use for prediction", features_without_churn, default=features_without_churn)
    
        # Update model training to use only selected features
        X = df[selected_features]
        if 'Churn' in df.columns:
            y = df['Churn']

        # Handle resampling, training, and prediction using the selected features
        if len(selected_features) > 0:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            prediction_mode = st.selectbox("Select Prediction Mode", ["Single Prediction", "Batch Prediction"])

            if prediction_mode == "Single Prediction":
                st.write("Enter the customer's information below:")
                user_input = {}
                cols = st.columns(2)

                for idx, column in enumerate(selected_features):
                    with cols[idx % 2]:
                        if column in ["tenure", "MonthlyCharges", "TotalCharges"]:
                            input_type = st.radio(
                                f"Input method for {column}",
                                ["Dropdown", "Manual"],
                                key=column
                            )
                            if input_type == "Dropdown":
                                unique_values = sorted(original_data[column].dropna().unique())
                                user_input[column] = st.selectbox(f"{column} (Dropdown)", unique_values)
                            else:
                                user_input[column] = st.number_input(
                                    f"{column} (Manual)",
                                    min_value=float(X[column].min()), 
                                    max_value=float(X[column].max()), 
                                    value=float(X[column].mean())
                                )
                        elif original_data[column].dtype == 'object' or column == 'SeniorCitizen':
                            options = original_data[column].unique()
                            user_input[column] = st.selectbox(f"{column}", options)
                            if column in label_encoders:
                                user_input[column] = label_encoders[column].transform([user_input[column]])[0]

                # Ensure all training columns are present in input data, add missing as default
                input_df = pd.DataFrame([user_input])
                missing_cols = set(selected_features) - set(input_df.columns)
                for col in missing_cols:
                    input_df[col] = 0  # Set default values for missing columns

                # Align input columns with training set
                input_df = input_df[selected_features]

                # Predict using the trained model
                if st.button("Predict"):
                    prediction = model.predict(input_df)[0]
                    prediction_label = "Churn" if prediction == 1 else "Not Churn"
                    st.write(f"The predicted outcome is: **{prediction_label}**")
                    if prediction_label == "Churn":
                        st.write("- rekomendasi jika churn.")
                    else:
                        st.write("- rekomendasi jika tidak churn.")

            elif prediction_mode == "Batch Prediction":
                st.write("Upload a dataset for batch prediction:")
                batch_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

                if batch_file is not None:
                    batch_data = pd.read_csv(batch_file)

                    # Preprocess batch data
                    for column in batch_data.select_dtypes(include=['object']).columns:
                        if column in label_encoders:
                            batch_data[column] = label_encoders[column].transform(batch_data[column])

                    # Perform predictions on the batch data
                    batch_predictions = model.predict(batch_data[selected_features])
                    batch_data['Churn_Prediction'] = ["Churn" if pred == 1 else "Not Churn" for pred in batch_predictions]

                    # Reverse the encoding for better interpretability
                    batch_data = reverse_encode(batch_data, label_encoders)

                    st.write("Batch Prediction Results:")
                    st.write(batch_data.head())

                    # Option to download the prediction results
                    csv = batch_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name='batch_predictions.csv',
                        mime='text/csv'
                    )

    elif option == "Data Overview":
        st.subheader("Data Overview")

        # Option to view original or preprocessed data
        view_option = st.selectbox("View data:", ["Original Data", "Preprocessed Data"])

        if view_option == "Original Data":
            st.write("### Original Data (Before Label Encoding)")
            st.write(original_data.head())

        elif view_option == "Preprocessed Data":
            st.write("### Preprocessed Data (After Label Encoding)")
            st.write(df.head())

        st.subheader("Data Preprocessing")
        if st.checkbox("Show summary statistics"):
            st.write(df.describe())

        # Visualization - Countplots for categorical columns
        st.subheader("Visualizations")
        st.write("Visualize the distribution of categorical columns:")
        categorical_columns = original_data.select_dtypes(include=['object']).columns.tolist()

        selected_column = st.selectbox("Select a categorical column for countplot", categorical_columns)

        if selected_column:
            fig, ax = plt.subplots()
            sns.countplot(data=original_data, x=selected_column, ax=ax)
            ax.set_title(f"Countplot of {selected_column}")
            st.pyplot(fig)

        # Visualization - Churn distribution for selected categorical columns
        st.subheader("Churn Distribution by Categorical Variable")
        st.write("Select a column to see the churn distribution:")
        selected_churn_column = st.selectbox("Select a categorical column for churn distribution", categorical_columns)

        if selected_churn_column:
            fig, ax = plt.subplots()
            sns.countplot(data=original_data, x=selected_churn_column, hue='Churn', ax=ax)
            ax.set_title(f"Churn Distribution by {selected_churn_column}")
            st.pyplot(fig)

    elif option == "Insights Dashboard":
        st.subheader("Insights Dashboard")
        st.write("Explore key insights from the customer churn data:")

        # Churn Rate
        churn_rate = (df['Churn'].sum() / df['Churn'].count()) * 100
        st.metric(label="Churn Rate", value=f"{churn_rate:.2f}%")

        # Average Tenure
        avg_tenure = df['tenure'].mean()
        st.metric(label="Average Tenure", value=f"{avg_tenure:.2f} months")

        # Monthly Charges Distribution
        st.subheader("Monthly Charges Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['MonthlyCharges'], kde=True, ax=ax)
        ax.set_title("Distribution of Monthly Charges")
        st.pyplot(fig)

        # 1. Churn by Contract Type with Percentage
        st.subheader("Churn by Contract Type")
        if 'Contract' in original_data.columns and 'Churn' in original_data.columns:
            contract_churn = original_data.groupby('Contract')['Churn'].mean().reset_index()
            contract_churn['Churn Rate (%)'] = contract_churn['Churn'] * 100
            st.write(contract_churn)
            
            # Visualize Churn by Contract Type
            fig, ax = plt.subplots()
            sns.countplot(data=original_data, x='Contract', hue='Churn', ax=ax)
            ax.set_title("Churn by Contract Type")
            st.pyplot(fig)

        # 2. Churn by Payment Method with Percentage
        st.subheader("Churn by Payment Method")
        if 'PaymentMethod' in original_data.columns and 'Churn' in original_data.columns:
            payment_churn = original_data.groupby('PaymentMethod')['Churn'].mean().reset_index()
            payment_churn['Churn Rate (%)'] = payment_churn['Churn'] * 100
            st.write(payment_churn)
            
            # Visualize Churn by Payment Method
            fig, ax = plt.subplots()
            sns.countplot(data=original_data, x='PaymentMethod', hue='Churn', ax=ax)
            ax.set_title("Churn by Payment Method")
            st.pyplot(fig)

        # 3. Churn by Internet Service Type with Percentage
        st.subheader("Churn by Internet Service Type")
        if 'InternetService' in original_data.columns and 'Churn' in original_data.columns:
            internet_churn = original_data.groupby('InternetService')['Churn'].mean().reset_index()
            internet_churn['Churn Rate (%)'] = internet_churn['Churn'] * 100
            st.write(internet_churn)

            # Visualize Churn by Internet Service Type
            fig, ax = plt.subplots()
            sns.countplot(data=original_data, x='InternetService', hue='Churn', ax=ax)
            ax.set_title("Churn by Internet Service Type")
            st.pyplot(fig)

        # 4. Tenure Distribution by Contract Type
        st.subheader("Tenure Distribution by Contract Type")
        if 'Contract' in original_data.columns:
            fig, ax = plt.subplots()
            sns.boxplot(data=original_data, x='Contract', y='tenure', ax=ax)
            ax.set_title("Tenure Distribution by Contract Type")
            st.pyplot(fig)

        # 5. Churn Rate by Senior Citizen
        st.subheader("Churn Rate by Senior Citizen")
        if 'SeniorCitizen' in original_data.columns:
            fig, ax = plt.subplots()
            sns.countplot(data=original_data, x='SeniorCitizen', hue='Churn', ax=ax)
            ax.set_title("Churn Rate by Senior Citizen")
            st.pyplot(fig)

        # 6. Average Monthly Charges by Payment Method
        st.subheader("Average Monthly Charges by Payment Method")
        if 'PaymentMethod' in original_data.columns:
            avg_monthly_charges = original_data.groupby('PaymentMethod')['MonthlyCharges'].mean().reset_index()
            st.write(avg_monthly_charges)

        # 7. Correlation between Tenure and Churn
        st.subheader("Correlation between Tenure and Churn")
        tenure_churn_corr = df['tenure'].corr(df['Churn'])
        st.write(f"Correlation between Tenure and Churn: {tenure_churn_corr:.2f}")

        # Adding CLV column to both df and original_data
        df['CLV'] = df['MonthlyCharges'] * df['tenure']  # Assuming CLV is simply monthly charges multiplied by tenure
        original_data['CLV'] = df['CLV']  # Add CLV to original_data as well
        
        # 8. Average CLV by Contract Type
        st.subheader("Average CLV by Contract Type")
        if 'Contract' in original_data.columns:
            avg_clv_by_contract = original_data.groupby('Contract')['CLV'].mean().reset_index()
            st.write(avg_clv_by_contract)

        # 9. Average CLV by Internet Service Type
        st.subheader("Average CLV by Internet Service Type")
        if 'InternetService' in original_data.columns:
            avg_clv_by_service = original_data.groupby('InternetService')['CLV'].mean().reset_index()
            st.write(avg_clv_by_service)

        # Export dataset with insights for Looker Studio
        st.subheader("Export Dataset with Insights")
        
        # Provide an option to download the data as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Dataset for Looker Studio",
            data=csv,
            file_name='churn_insights_with_clv.csv',
            mime='text/csv'
        )
