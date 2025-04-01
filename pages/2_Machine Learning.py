import time
from memory_profiler import memory_usage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

st.set_page_config(page_title="Hypertension Prediction & Model Performance", layout="wide")

tab1, tab2, tab3 = st.tabs(["Machine Learning", "Hyperparameter Tuning", "Principal Component Analysis"])

with tab1:
    # Title
    st.title("Hypertension Classification & Model Performance Application")

    # Đường dẫn đến file cố định
    file_path = r"./dataset/raw/PPG-BP dataset.xlsx"

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

            # # Show detailed counts of predictions
            # st.subheader("Prediction Details")
            # details_df = pd.DataFrame({
            #     "True Class": y_test,
            #     "Predicted Class": y_pred
            # })
            # st.write(details_df)

            # Visualize feature pairs
            st.subheader("Visualization of Feature Pairs")
            df_copy = df.copy()
            feature_pairs = [
                ('Systolic BP(mmHg)', 'Diastolic BP(mmHg)')
            ]

            for feature1, feature2 in feature_pairs:
                st.write(f"Visualizing with features: {feature1}, {feature2}")

                X_train, X_test, y_train, y_test = train_test_split(df_copy[[feature1, feature2]], y, test_size=0.3, random_state=42, stratify=y)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Retrain model with new feature pair
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                df_test_copy = X_test.copy()
                df_test_copy['True Label'] = y_test.values
                df_test_copy['Predicted Label'] = y_pred

                # Plot true label clustering
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                sns.scatterplot(data=df_test_copy, x=feature1, y=feature2, hue='True Label', palette='Set2')
                plt.title(f'{model_name} - True Label Clustering')

                # Plot predicted label clustering
                plt.subplot(1, 2, 2)
                sns.scatterplot(data=df_test_copy, x=feature1, y=feature2, hue='Predicted Label', palette='Set2')
                plt.title(f'{model_name} - Predicted Label Clustering')

                # Highlight misclassified points
                misclassified = df_test_copy[df_test_copy['True Label'] != df_test_copy['Predicted Label']]

                for i in range(misclassified.shape[0]):
                    plt.gca().add_patch(patches.Ellipse(
                        (misclassified[feature1].iloc[i], misclassified[feature2].iloc[i]),
                        width=3, height=2, edgecolor='red', facecolor='none', lw=1.5))

                plt.tight_layout()
                st.pyplot(plt)

            # Compare models button for performance tracking   
            if st.button("Compare Models (Performance Metrics)"):
                st.image("./images/images_gui/ML/Accurac_compare.png", use_container_width=True)
                st.image("./images/images_gui/ML/Train_time.png", use_container_width=True)
                st.image("./images/images_gui/ML/Train_memory.png", use_container_width=True)
    else:
        st.write("Click the button in the sidebar to load the dataset.")

with tab2:
    st.title("Hyperparameter Tuning for SVM")

    # Define the parameter grid
    param_grid = {
        'C': [0.1, 1, 10],          
        'gamma': [0.01, 0.02, 0.03],  
        'kernel': ['rbf', 'linear']  # Linear vì scatter plot predict và true label nằm gần một mặt phẳng
    }

    # Check if dataset is loaded
    if st.session_state.dataset_loaded:
        df = st.session_state.data

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

        target = df['Hypertension']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df[['Systolic BP(mmHg)', 'Diastolic BP(mmHg)']],
            target, test_size=0.3, random_state=42, stratify=target
        )

        # Standardize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Perform GridSearchCV
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
        grid.fit(X_train_scaled, y_train)

        # Display best parameters
        st.subheader("Best Hyperparameters")
        st.write(grid.best_params_)

        # Predict with tuned model
        y_pred = grid.best_estimator_.predict(X_test_scaled)

        # Accuracy
        accuracy_tuned = accuracy_score(y_test, y_pred)
        st.subheader("Model Accuracy After Tuning")
        st.markdown(f'<p style="color:red;font-size:18px;">Accuracy: <strong>{accuracy_tuned:.2f}</strong></p>', unsafe_allow_html=True)

        # Classification report
        st.subheader("Classification Report After Tuning")
        report = classification_report(y_test, y_pred, output_dict=True, target_names=["Normal", "Stage 1 Hypertension", "Stage 2 Hypertension", "Prehypertension"])
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Confusion matrix
        st.subheader("Confusion Matrix After Tuning")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        cax = ax.matshow(cm, cmap="Reds")  # Sử dụng màu sắc giống Tab 1
        plt.colorbar(cax)
        plt.title("Confusion Matrix: SVM with Hyperparameter Tuning", pad=20)
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Add numerical values to confusion matrix
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', color="black" if cm[i, j] < cm.max() / 2 else "white")

        st.pyplot(fig)

        st.image("./images/images_gui/ML/SVM_Tuning_Accuracy.png", caption="Distribution by Hypertension")
        st.image("./images/images_gui/ML/SVM_Tuning_Cluster.png", caption="Distribution by Hypertension", use_container_width=True)
    else:
        st.write("Please load the dataset in Machine Learning Tab to perform hyperparameter tuning.")

with tab3:
    st.title("Principal Component Analysis (PCA)")

    if st.session_state.dataset_loaded:
        df = st.session_state.data

        # Xác định và hiển thị các đặc trưng số
        numerical_features = [feature for feature in df.columns if df[feature].dtypes != object]

        if len(numerical_features) > 0:
            # Chuẩn hóa dữ liệu
            X = df[numerical_features].dropna()  # Loại bỏ hàng có giá trị NaN nếu có
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Thực hiện PCA
            pca = PCA()
            pca.fit(X_scaled)

            # Lấy explained variance ratio và cumulative explained variance ratio
            explained_variance_ratio = pca.explained_variance_ratio_
            cum_explained_variance = np.cumsum(explained_variance_ratio)

            # Tạo bảng
            variance_table = pd.DataFrame({
                "Principal Component": [f"PC{i+1}" for i in range(len(explained_variance_ratio))],
                "Explained Variance Ratio": explained_variance_ratio,
                "Cumulative Variance Ratio": cum_explained_variance
            })

            # Hiển thị bảng
            st.write("### Explained Variance and Cumulative Variance")
            st.dataframe(variance_table.style.format({
                "Explained Variance Ratio": "{:.4f}",
                "Cumulative Variance Ratio": "{:.4f}"
            }), use_container_width=True)

            # Xác định số lượng thành phần chính giữ lại 90% biến thiên
            n_components = np.argmax(cum_explained_variance >= 0.90) + 1
            st.write(f"Number of Principal Components to retain 90% variance: **{n_components}**")

            st.write(
                f"To retain at least 90% of the original data information:\n"
                f"- We need to keep **{n_components} components**.\n"
                f"- This ensures we preserve **{cum_explained_variance[n_components - 1]:.2%}** "
                f"of the variance.\n"
                f"- The remaining components contribute very little and can be discarded."
            )

            # Giảm chiều dữ liệu với số thành phần chính đã chọn
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)

            # Hiển thị kích thước dữ liệu sau khi giảm chiều
            st.write(f"Dataset shape after PCA: {X_pca.shape}")

            # Train-test split
            y = df['Hypertension']
            X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)

            # So sánh các mô hình sau PCA
            st.subheader("Model Performance After PCA")
            accuracy_dict_pca = {}

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Tính toán accuracy
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_dict_pca[model_name] = accuracy

                # Lưu Classification Report và Confusion Matrix cho từng mô hình
                st.session_state[f"{model_name}_accuracy"] = accuracy
                st.session_state[f"{model_name}_report"] = classification_report(y_test, y_pred, target_names=["Normal", "Stage 1 Hypertension", "Stage 2 Hypertension", "Prehypertension"], output_dict=True)
                st.session_state[f"{model_name}_cm"] = confusion_matrix(y_test, y_pred)

            # Hiển thị danh sách các mô hình để người dùng chọn
            selected_model = st.selectbox("Choose a model to view the Classification Report and Confusion Matrix:", list(models.keys()))

            # Hiển thị báo cáo chi tiết cho mô hình được chọn
            if selected_model:
                st.write(f"### {selected_model}")
                
                # Hiển thị Accuracy màu đỏ
                accuracy = st.session_state[f"{selected_model}_accuracy"]
                st.markdown(
                    f'<p style="color:red;font-size:18px;">Accuracy: <strong>{accuracy:.2f}</strong></p>',
                    unsafe_allow_html=True
                )

                # Hiển thị Classification Report
                report_df = pd.DataFrame(st.session_state[f"{selected_model}_report"]).transpose()
                st.dataframe(report_df)

                # Vẽ confusion matrix cho mô hình được chọn
                cm = st.session_state[f"{selected_model}_cm"]
                fig, ax = plt.subplots()
                cax = ax.matshow(cm, cmap="Reds")
                plt.colorbar(cax)
                plt.title(f"Confusion Matrix: {selected_model}", pad=20)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', color="black" if cm[i, j] < cm.max()/2 else "white")
                st.pyplot(fig)

            # Tổng hợp và so sánh độ chính xác của các mô hình
            st.subheader("Model Comparison After PCA")
            accuracy_df = pd.DataFrame.from_dict(accuracy_dict_pca, orient='index', columns=["Accuracy"]).sort_values(by="Accuracy", ascending=False)
            st.bar_chart(accuracy_df)
            st.image("./images/images_gui/ML/PCA_accuracy.png")
        else:
            st.error("No numerical features available in the dataset.")
    else:
        st.write("Please load the dataset in Machine Learning Tab to perform PCA.")