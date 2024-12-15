import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

# Title of the app
st.title("Introduction: Hypertension Classification and Prediction Using Machine Learning")

# Introduction Section
st.write("""
### Introduction
Metabolic diseases are extremely common:\n
    - Hyperglycemia\n
    - Diabetes\n
    - Stroke\n
    - Heart Disease\n
According to the World Health Organization (WHO), the top death causes in **Vietnam** and **World** are:
""")
st.write("""
    ### <span style='color:red;'>- Stroke</span>
    ### <span style='color:red;'>- Heart Disease</span>
    """, unsafe_allow_html=True)
# Button to show the causes
if st.button("Show Data"):
    # Display images
    try:
        st.image("../../images/images_gui/Intro/death_cause_vn.png", caption="Top Causes of Death in Vietnam")
        st.image("../../images/images_gui/Intro/world.png", caption="Top Causes of Death Globally")

        st.write('**Hypertension** is the main cause!')
        st.write('-> Reduce **Hypertension** might reduce mortality.')
    except FileNotFoundError:
        st.error("Images are missing.")