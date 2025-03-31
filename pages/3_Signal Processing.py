import streamlit as st
import os
import pandas as pd

# Đường dẫn các thư mục và file
PPG_SQI_file = '../../results/PPG_SQI_Results.xlsx'
folder_0_subject = '../../dataset/raw/0_subject'
folder_0_subjectSQI = '../../results/0_subjectSQI'
images_signal = '../../images/images_signal'
images_spectrum = '../../images/images_spectrum'
folder_normalized = '../../dataset/normalize/0_subjectSQI_normalized'

st.title("PPG Waveform Signal Processing")

# Hàm hiển thị nội dung thư mục (dành cho file .txt)
def show_folder_content(folder_path, folder_title):
    if os.path.exists(folder_path):
        st.subheader(f"Files in {folder_title}")
        
        # Lấy danh sách file trong thư mục
        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        
        if len(txt_files) == 0:
            st.warning(f"No TXT files found in the folder `{folder_path}`.")
        else:
            # Chọn file từ danh sách file
            selected_file = st.selectbox(f"Files in `{folder_title}`:", txt_files, key=folder_path)

            # Khởi tạo trạng thái hiển thị nếu chưa có
            if f"show_content_{folder_title}" not in st.session_state:
                st.session_state[f"show_content_{folder_title}"] = False

            # Nút show/hide
            if st.button(f"Show/Hide Content ({folder_title})", key=f"toggle_{folder_title}"):
                # Toggle trạng thái hiển thị
                st.session_state[f"show_content_{folder_title}"] = not st.session_state[f"show_content_{folder_title}"]

            # Hiển thị hoặc ẩn nội dung file
            if st.session_state[f"show_content_{folder_title}"]:
                file_path = os.path.join(folder_path, selected_file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                st.write(f"### Content of `{selected_file}`:")
                st.text(content)
    else:
        st.error(f"The folder `{folder_path}` does not exist. Please check the path.")

# Hàm hiển thị nội dung thư mục ảnh
def show_image_folder_with_subfolders(folder_path):
    if os.path.exists(folder_path):
        st.subheader("Select an Image from Signal Subfolders")

        # Lấy danh sách các thư mục con
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        subfolders = sorted(subfolders)  # Sắp xếp thư mục con theo thứ tự từ bé đến lớn

        # Chọn thư mục con
        selected_subfolder = st.selectbox("Select a subfolder:", subfolders)

        if selected_subfolder:
            # Lấy đường dẫn đến thư mục con được chọn
            subfolder_path = os.path.join(folder_path, selected_subfolder)

            # Lấy danh sách file ảnh trong thư mục con
            image_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith((".png"))])

            if len(image_files) == 0:
                st.warning(f"No image files found in the folder `{subfolder_path}`.")
            else:
                # Chọn file ảnh
                selected_image = st.selectbox(f"Select an image from `{selected_subfolder}`:", image_files)

                # Hiển thị hình ảnh đã chọn
                if selected_image:
                    image_path = os.path.join(subfolder_path, selected_image)
                    st.image(image_path, caption=f"Selected Image: {selected_image}", use_container_width=True)
    else:
        st.error(f"The folder `{folder_path}` does not exist. Please check the path.")

# Hàm hiển thị các data point lớn nhất và nhỏ nhất từ thư mục normalized
def show_normalized_data_points(folder_path):
    if os.path.exists(folder_path):
        st.subheader("Normalized Data Points Overview")

        # Lấy danh sách file .txt
        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        txt_files = sorted(txt_files)

        if len(txt_files) == 0:
            st.warning(f"No TXT files found in the folder `{folder_path}`.")
        else:
            # Chọn file để xem
            selected_file = st.selectbox("Select a normalized file:", txt_files, key="normalized")

            file_path = os.path.join(folder_path, selected_file)

            # Đọc file và lấy giá trị lớn nhất, nhỏ nhất
            try:
                data = pd.read_csv(file_path, header=None)  # Giả sử dữ liệu không có header
                max_values = data.max()
                min_values = data.min()

                # Tạo bảng hiển thị min và max
                min_max_table = pd.DataFrame({
                    "Statistic": ["Min Value", "Max Value"],
                    **{f"Column {i+1}": [min_values[i], max_values[i]] for i in range(len(min_values))}
                })

                # Hiển thị bảng min và max
                st.write(f"### File: {selected_file}")
                st.table(min_max_table)

            except Exception as e:
                st.error(f"Error reading file `{selected_file}`: {e}")
    else:
        st.error(f"The folder `{folder_path}` does not exist. Please check the path.")

st.write(
    "There are **219** patients:\n"
    "- **Each patient** has **3 signal segments**.\n"
    "- There are **657 signal files** as total.\n"
    "- Find best segment along each patient."
)

# Hiển thị nội dung của folder_0_subject
show_folder_content(folder_0_subject, "0_subject")

# Hiển thị file PPG_SQI_file
st.subheader("PPG SQI Results")
if os.path.exists(PPG_SQI_file):
    st.image("../../images/images_gui/Signal/SQI_formula.png", caption="SQI Formula")
    df = pd.read_excel(PPG_SQI_file)
    st.dataframe(df, use_container_width=True)
else:
    st.error(f"The file `{PPG_SQI_file}` does not exist. Please check the path.")

# Hiển thị nội dung của folder_0_subjectSQI
show_folder_content(folder_0_subjectSQI, "0_subjectSQI")

# Normalize Section
st.subheader('Normalize')
st.write(
    "- Global max amplitute: 4011\n"
    "- Global min amplitute: 1394\n"
)
show_normalized_data_points(folder_normalized)

show_image_folder_with_subfolders(images_signal)
st.write("""
2 main phases:\n
- Rising phase: heart **contracts** → blood flow **increases**\n
- Falling phase: heart **relaxes** → blood flow **decreases**\n
- Time: 2.1s\n
-> Images are almost the same, no obvious difference. Need another method for insight\n
""")

show_image_folder_with_subfolders(images_spectrum)
st.write("""
- FFT: Time-Domain -> Frequency-Domain\n
-> **High frequency** at: **1-2 Hz**. The rest have low amplitude\n
- PSD: Power density with frequency\n
-> **Power decreases** as **frequency increases**\n
- Spectrogram: Energy change with time and frequency\n
-> **Low frequency energy (0–50 Hz)**, and is stable over time\n
""")
st.markdown(
    '<p style="color:red;font-size:18px;">-> Signal is not clear enough. Since it not clearly distinguishing between stages of hypertension. </p>',
    unsafe_allow_html=True
)