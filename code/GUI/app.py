import time
from memory_profiler import memory_usage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Set page title
st.set_page_config(page_title="Hypertension Prediction & Model Performance", layout="wide")

# Title
st.title("Hypertension Classification & Model Performance Application")

# Đường dẫn đến file cố định
file_path = r"C:\PPG_BP_Code\dataset\raw\PPG-BP dataset - Processed.xlsx"

# Load data
st.sidebar.header("Load Dataset")

if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False
    st.session_state.data = None

if st.sidebar.button("Load Dataset"):
    try:
        # Load dataset từ đường dẫn cố định
        df = pd.read_excel(file_path)
        st.session_state.dataset_loaded = True
        st.session_state.data = df

    except Exception as e:
        st.error(f"Error loading dataset: {e}")

if st.session_state.dataset_loaded:
    df = st.session_state.data

    # Display dataset
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Preprocess dataset
    df = df.rename(columns={
        'Systolic Blood Pressure(mmHg)': 'Systolic BP(mmHg)',
        'Diastolic Blood Pressure(mmHg)': 'Diastolic BP(mmHg)'
    })

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

    if df['Hypertension'].dtype == object:
        df['Hypertension'] = df['Hypertension'].apply(convert_hypertension_status)

    df['Hypertension'] = df['Hypertension'].astype('int')

    # Show summary of hypertension statuses
    st.subheader("Summary of Hypertension Status")
    status_counts = df['Hypertension'].value_counts().sort_index()
    status_labels = ["Normal", "Stage 1 Hypertension", "Stage 2 Hypertension", "Prehypertension"]
    summary_df = pd.DataFrame({
        "Status": status_labels,
        "Count": status_counts
    })
    st.table(summary_df)

    # Select features and target
    feature_columns = st.multiselect(
        "Select Feature Columns", 
        [col for col in df.columns if col != 'Hypertension'],
        default=['Systolic BP(mmHg)', 'Diastolic BP(mmHg)']
    )

    if len(feature_columns) > 0:
        target_column = 'Hypertension'

        # Split data
        X = df[feature_columns]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # Standardize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "SVM": SVC(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

        model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
        model = models[model_name]

        # Train and predict
        if model_name in ["Logistic Regression", "SVM", "K-Nearest Neighbors", "Naive Bayes"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Display results
        st.subheader("Model Results")
        st.write(f"Selected Model: **{model_name}**")

        accuracy = accuracy_score(y_test, y_pred)
        st.markdown(f'<p style="color:red;font-size:18px;">Accuracy: <strong>{accuracy:.2f}</strong></p>', unsafe_allow_html=True)

        st.subheader("Detailed Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True, target_names=status_labels)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cax = ax.matshow(cm, cmap="Reds")
        plt.colorbar(cax)
        plt.title(f"Confusion Matrix: {model_name}", pad=20)
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Add numerical values to the cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', color="black" if cm[i, j] < cm.max()/2 else "white")

        st.pyplot(fig)

        # Show detailed counts of predictions
        st.subheader("Prediction Details")
        details_df = pd.DataFrame({
            "True Class": y_test,
            "Predicted Class": y_pred
        })
        st.write(details_df)

        # Compare models button for performance tracking   
        if st.button("Compare Models (Performance Metrics)"):
            time_dict = {}
            memory_dict = {}
            accuracy_dict = {}
            feature_sets = [feature_columns]

            for features in feature_sets:
                for model_name, model in models.items():
                    start_time = time.time()
                    mem_usage = memory_usage((model.fit, (X_train_scaled, y_train)), max_iterations=1)
                    end_time = time.time()

                    elapsed_time = end_time - start_time
                    max_memory = max(mem_usage)
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)

                    if model_name not in time_dict:
                        time_dict[model_name] = []
                        memory_dict[model_name] = []
                        accuracy_dict[model_name] = []

                    time_dict[model_name].append(elapsed_time)
                    memory_dict[model_name].append(max_memory)
                    accuracy_dict[model_name].append(accuracy)

            # Plot performance
            st.subheader("Model Performance (Time, Memory, and Accuracy)")

            def plot_performance(time_dict, memory_dict, accuracy_dict, feature_sets):
                models = list(time_dict.keys())
                index = np.arange(len(feature_sets))
                bar_width = 0.2

                # Plot time
                fig, ax = plt.subplots(figsize=(10, 5))
                for i, model_name in enumerate(models):
                    ax.bar(index + i * bar_width, [time_dict[model_name][0]], bar_width, label=model_name)
                ax.set_xlabel("Feature Sets")
                ax.set_ylabel("Time (Seconds)")
                ax.set_title("Training Time by Models")
                ax.set_xticks(index + bar_width * (len(models) - 1) / 2)
                ax.set_xticklabels([f"{len(f)} features" for f in feature_sets])
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                st.pyplot(fig)

                # Plot memory
                fig, ax = plt.subplots(figsize=(10, 5))
                for i, model_name in enumerate(models):
                    ax.bar(index + i * bar_width, [memory_dict[model_name][0]], bar_width, label=model_name)
                ax.set_xlabel("Feature Sets")
                ax.set_ylabel("Memory Usage (MiB)")
                ax.set_title("Memory Usage by Models")
                ax.set_xticks(index + bar_width * (len(models) - 1) / 2)
                ax.set_xticklabels([f"{len(f)} features" for f in feature_sets])
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                st.pyplot(fig)

                # Plot accuracy
                fig, ax = plt.subplots(figsize=(14, 8))
                for i, model_name in enumerate(models):
                    bars = ax.bar(index + i * bar_width, accuracy_dict[model_name], bar_width, label=model_name)
                    for bar in bars:
                        value = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.2f}', ha='center', va='bottom')
                ax.set_xlabel("Feature Sets")
                ax.set_ylabel("Accuracy")
                ax.set_title("Accuracy of Models with Varying Feature Sets")
                ax.set_xticks(index + bar_width * (len(models) - 1) / 2)
                ax.set_xticklabels([f"{len(f)} features" for f in feature_sets])
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                st.pyplot(fig)

            plot_performance(time_dict, memory_dict, accuracy_dict, feature_sets)

else:
    st.write("Click the button in the sidebar to load the dataset.")
