import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Đường dẫn model và file dữ liệu
model_path = r"C:\PPG_BP_Code\save_models\MLP_model.keras"
PPG_file = r"../../dataset/raw/PPG-BP dataset.xlsx"

# Tiêu đề ứng dụng
st.title("MLP Model Hyperparameters, Loss & Accuracy")

# **1. Load và Hiển Thị Siêu Tham Số Model**
try:
    # Load model Keras
    model = load_model(model_path)
    st.success("MLP model loaded successfully!")

except Exception as e:
    st.error(f"Error loading model: {e}")

# **2. Load Dữ Liệu**
@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path)
    
    # Chuyển đổi giá trị 'Hypertension'
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

    if data['Hypertension'].dtype == object:
        data['Hypertension'] = data['Hypertension'].apply(convert_hypertension_status)

    # Chọn các feature
    selected_features_7 = ['Age(year)', 'Height(cm)', 'Weight(kg)',
                           'Systolic Blood Pressure(mmHg)', 'Diastolic Blood Pressure(mmHg)',
                           'Heart Rate(b/m)', 'BMI(kg/m^2)', 'Hypertension']
    data_selected = data[selected_features_7].copy()
    data_selected['Hypertension'] = data_selected['Hypertension'].astype('int')
    return data_selected

# Load dữ liệu
data = load_data(PPG_file)

# Hiển thị dữ liệu ban đầu
st.write("### Dataset Preview")
st.write(data.head())

# **3. Tạo Nút Predict**
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False

def toggle_predict():
    st.session_state.predict_clicked = not st.session_state.predict_clicked

# Nút Predict
st.button("Predict", on_click=toggle_predict)

# **4. Tiến Hành Dự Đoán và Hiển Thị Kết Quả**
if st.session_state.predict_clicked:
    try:
        # Chia dữ liệu thành X và y
        X = data.drop(columns=['Hypertension'])
        y = data['Hypertension']

        # Chia dữ liệu thành tập train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Tính loss và predict
        loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
        y_pred = model.predict(X_test_scaled)
        y_pred_labels = np.argmax(y_pred, axis=1)

        # Hiển thị Loss và Accuracy
        st.markdown(
            "<h3>Multilayer perceptron (MLP) Prediction</h3>"
            f"<p style='color:green; font-size:20px;'><strong>Accuracy: {accuracy:.2f}</strong></p>"
            f"<p style='color:red; font-size:20px;'><strong>Loss: {loss:.2f}</strong></p>",
            unsafe_allow_html=True,
        )

        # Hiển thị Classification Report
        st.write("### Classification Report")
        report = classification_report(y_test, y_pred_labels, target_names=["Normal", "Stage 1", "Stage 2", "Prehypertension"], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.table(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

        # Hiển thị hình ảnh
        st.image('../../images/images_gui/DL/MLP_accuracy.png', caption="Training and Validation Accuracy")
        st.image('../../images/images_gui/DL/MLP_lost.png', caption="Training and Validation Loss")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
