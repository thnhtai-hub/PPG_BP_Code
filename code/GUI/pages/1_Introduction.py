import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

# Title of the app
st.title("PPG-BP Dataset Introduction")

# Introduction Section
st.write("""
### Introduction
According to the World Health Organization (WHO), the 1st and 2nd death causes in Vietnam are:
- **Stroke.**
- **Heart Disease.**

Globally, the 1st and 3rd death causes are:

""")

# Button to show the causes
if st.button("..."):
    st.write("""
    - ### Stroke
    - ### Heart Disease
    """)

    # Display images
    try:
        st.image("../../images/images_gui/death_cause_vn.png", caption="Top Causes of Death in Vietnam")
        st.image("../../images/images_gui/world.png", caption="Top Causes of Death Globally")
        st.image("../../images/images_gui/hypertension.jpg", caption="Hypertension")
    except FileNotFoundError:
        st.error("Images are missing.")