import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt

# Title of the app
st.title("PPG-BP Dataset Viewer with Data Preprocessing")

# Sidebar section
st.sidebar.title("Options")
load_data_button = st.sidebar.button("Load Data")  # Add button to sidebar

# Load Data if button is clicked
if load_data_button:
    try:
        # Load and preprocess data
        df = pd.read_excel('../../dataset/raw/PPG-BP dataset.xlsx')

        # Rename columns for better readability
        df = df.rename(columns={
            'Systolic Blood Pressure(mmHg)': 'Systolic BP(mmHg)',
            'Diastolic Blood Pressure(mmHg)': 'Diastolic BP(mmHg)'
        })

        # Display the data in full size
        st.write("### Data Loaded Successfully!")
        st.dataframe(df, use_container_width=True)  # Expand table to full width
        st.write(f"Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}")

        st.write("""
        ### Steps to do
        1. Dataset Overview
        2. Missing value
        3. All the Numerical Variables
        4. Categorical Variables Encoding
        5. Visualization
        """)

        # 1. Dataset Overview
        st.write("# 1. Dataset Overview")
        st.write("## 1.1 Data Types")
        st.write(df.dtypes)
        st.write("-> Identify Numerical Data and Categorical Data")
        st.write("## 1.2 Descriptive Statistics")
        st.write(df.describe().T)
        st.write("Describe and Summarize data")

        # 2. Missing value
        # Check for missing values before handling
        missing_before = df.isnull().sum()

        # Handle categorical missing values
        def replace_cat_feature(df, features_nan):
            data = df.copy()
            data[features_nan] = data[features_nan].fillna('Missing')
            return data

        # Identify categorical features with missing values
        features_nan = [feature for feature in df.columns if df[feature].isnull().sum() > 0 and df[feature].dtypes == 'object']
        df = replace_cat_feature(df, features_nan)

        # Check for missing values after handling
        missing_after = df.isnull().sum()

        # Visualization for Missing Values
        st.write("# 2. Missing Values: Before vs After Handling")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # Chart 1: Before Handling
        axes[0].barh(missing_before.index, missing_before, color='salmon')
        axes[0].set_title("Before Handling Missing Values")
        axes[0].set_xlabel("Number of Missing Values")
        axes[0].set_ylabel("Features")

        # Chart 2: After Handling
        axes[1].barh(missing_after.index, missing_after, color='seagreen')
        axes[1].set_title("After Handling Missing Values")
        axes[1].set_xlabel("Number of Missing Values")

        # Adjust layout
        plt.tight_layout()
        st.pyplot(fig)

        # Numerical Features Overview
        numerical_features = [feature for feature in df.columns if df[feature].dtypes != object and feature not in ['Num.', 'subject_ID']]
        st.write("# 3. Numerical Features Overview")
        st.dataframe(df[numerical_features].head(), use_container_width=True)
        st.write(f"Number of numerical features: {len(numerical_features)}")

        # Convert hypertension status
        def convert_hypertension_status(status):
            if status == "Normal":
                return 0
            elif status == "Stage 1 hypertension":
                return 1
            elif status == "Stage 2 hypertension":
                return 2
            elif status == "Prehypertension":
                return 3
            else:
                return status

        # Apply conversion if 'Hypertension' exists and is object dtype
        if 'Hypertension' in df.columns:
            if df['Hypertension'].dtype == object:
                df['Hypertension'] = df['Hypertension'].apply(convert_hypertension_status)

            # Ensure 'Hypertension' is an integer type
            df['Hypertension'] = df['Hypertension'].astype('int')

        # Display processed data
        st.write("# 4. Encoding Hypertension into label")
        st.dataframe(df.head(), use_container_width=True)
        st.write("""
        - **Normal**: 0
        - **Stage 1 Hypertension**: 1
        - **Stage 2 Hypertension**: 2
        - **Prehypertension**: 3
        """)


        # Display Images
        st.write("# 5. Visualization")
        st.write("## 5.1 Distribution by Hypertension")
        try:
            st.image("../../images/images_gui/distribution_by_hypertension.png", caption="Distribution by Hypertension", use_container_width=True)
        except FileNotFoundError:
            st.warning("The file '../../images/images_gui/distribution_by_hypertension.png' was not found.")

        st.write("## 5.2 Multivariate Analysis")
        try:
            st.image("../../images/images_gui/pair_plot_hue.png", caption="Pair Plot with Hue", use_container_width=True)
        except FileNotFoundError:
            st.warning("The file '../../images/images_gui/pair_plot_hue.png' was not found.")

    except FileNotFoundError:
        st.sidebar.error("The file 'PPG-BP dataset.xlsx' was not found. Please check the file path.")
    except Exception as e:
        st.sidebar.error(f"An error occurred: {e}")


        

