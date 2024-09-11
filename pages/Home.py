import streamlit as st

st.title(":battery: BattImpact")

st.write("Welcome to BattImpact, a project that aims to help key stakeholders wthin the battery industry understand how to make long lasting and effective strategic decision to secure sustainable and earth friendly supply chains.")




col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Data Exploration", use_container_width=True):
        st.switch_page("pages/Data Exploration.py")
with col2:
    if st.button("Results", use_container_width=True):
        st.switch_page("pages/Results.py")
with col3:
    if st.button("Battery Viewer", use_container_width=True):
        st.switch_page("pages/BatteryViewer.py")


st.divider()

st.markdown("**How it works**")

st.markdown("""
BattImpact uses advanced machine learning models to analyze and predict the performance and impact of different battery chemistries. 
By leveraging large datasets from the Materials Project, we can train our models to understand the relationships between 
battery properties and their environmental, economic, and social impacts.

""")

st.markdown("""
            **Understanding Our Results**
            """)
documentation_options = ['RFECV Results', 'Learning Curves', 'Parity Plots', 'Intrinsic Feature Importances', 'SHAP Network Plot', 'SHAP Interaction Plot', 'Force Plots', 'Waterfall Plots', 'Linear Regression Coefficient Information', 'XGBoostHyperparameters']

documentation_choice = st.selectbox("Select a documentation option", documentation_options, key="documentation_choice")


    