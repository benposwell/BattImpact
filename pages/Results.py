import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from supabase import create_client, Client

feature_dictionary = {
    'Structural Encoding': ['Li_formula_discharge', 'C_formula_discharge', 'In_formula_discharge', 
                            'Bi_formula_discharge', 'Na_formula_discharge', 'Tl_formula_discharge', 
                            'Sb_formula_discharge', 'K_formula_discharge', 'Rb_formula_discharge', 
                            'Mg_formula_discharge', 'Mn_formula_discharge', 'O_formula_discharge', 
                            'Ca_formula_discharge', 'Nb_formula_discharge', 'S_formula_discharge', 
                            'Co_formula_discharge', 'Al_formula_discharge', 'Cu_formula_discharge', 
                            'Zn_formula_discharge', 'Ni_formula_discharge', 'Ti_formula_discharge', 
                            'As_formula_discharge', 'Cs_formula_discharge', 'Sn_formula_discharge', 
                            'Sc_formula_discharge', 'Si_formula_discharge', 'P_formula_discharge', 
                            'Mo_formula_discharge', 'Cr_formula_discharge', 'V_formula_discharge', 
                            'Ge_formula_discharge', 'N_formula_discharge', 'Fe_formula_discharge', 
                            'Pd_formula_discharge', 'Y_formula_discharge', 'Ga_formula_discharge', 
                            'Pt_formula_discharge', 'Te_formula_discharge', 'Se_formula_discharge', 
                            'F_formula_discharge', 'W_formula_discharge', 'Ho_formula_discharge', 
                            'Ba_formula_discharge', 'Be_formula_discharge', 'La_formula_discharge', 
                            'Sr_formula_discharge', 'Re_formula_discharge', 'Ta_formula_discharge', 
                            'Pr_formula_discharge', 'Ir_formula_discharge', 'Cl_formula_discharge', 
                            'I_formula_discharge', 'Lu_formula_discharge', 'Tb_formula_discharge', 
                            'Tm_formula_discharge', 'Er_formula_discharge', 'Ag_formula_discharge', 
                            'Zr_formula_discharge', 'Dy_formula_discharge', 'Cd_formula_discharge', 
                            'H_formula_discharge', 'Br_formula_discharge', 'Ce_formula_discharge', 
                            'B_formula_discharge', 'Tc_formula_discharge', 'Rh_formula_discharge', 
                            'Nd_formula_discharge', 'U_formula_discharge', 'Gd_formula_discharge', 
                            'Ru_formula_discharge', 'Au_formula_discharge', 'Hg_formula_discharge', 
                            'Sm_formula_discharge', 'Hf_formula_discharge', 'Yb_formula_discharge', 
                            'Pb_formula_discharge', 'Eu_formula_discharge'], 
    'Battery Properties': ['average_voltage', 'capacity_grav', 'energy_grav', 'max_delta_volume', 
                           'working_ion_Al', 'working_ion_Ca', 'working_ion_Cs', 'working_ion_K', 
                           'working_ion_Li', 'working_ion_Mg', 'working_ion_Na', 'working_ion_Rb', 
                           'working_ion_Y', 'working_ion_Zn'], 
    'Environmental Impact Features': ['ADP (Kg)', 'CCH', 'ODP', 'HT', 'POF', 'PM', 'IR', 'CCE', 'TA', 
                                      'FE', 'TET', 'FET', 'MET', 'ALO', 'ULO', 'NLT', 'Human Health', 
                                      'Eco- systems', 'Criticality EI Score'], 
    'Socioeconomic Impact Features': ['Political Stability', 'Demand growth', 'Mining capacity', 
                                      'Concentration of reserves', 'Concentration of production', 
                                      'Trade barriers', 'Feasability of exploration projects', 
                                      'Price volatility', 'Occurence of co-production', 'Primary material use', 
                                      'Company concentration', '(Non) compliance with social standards']
}

target_list = [
    '(Non) compliance with social standards',
    'Demand growth',
    'EU_Critical',
    'US_Critical',
    'UK_Critical',
    'ADP (Kg)',
    'Criticality EI Score',
    'Human Health',
    'Eco- systems',
    'CCH', 'ODP', 'HT', 
    'POF', 'PM', 'IR', 
    'CCE', 'TA', 'FE', 
    'TET', 'FET', 'MET', 
    'ALO', 'ULO', 'NLT',
    'average_voltage',
    'capacity_grav',
    'energy_grav',
    'Price (latest, 1998)'
]

@st.cache_resource
def init_connection():
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_data(table_name, column_name ='*'):
    supabase = init_connection()
    response = supabase.table(table_name).select(column_name).execute()
    return response.data

if "regression_table" not in st.session_state:
    st.session_state.regression_table = None
if "xgboost_table" not in st.session_state:
    st.session_state.xgboost_table = None

def display_linear_regression_visuals(supabase, result):
    st.subheader("RFECV Results")
    rfecv_result = supabase.table('visualizations').select('*').eq('model_id', result['id']).eq('vis_type', 'rfecv').execute()
    if rfecv_result.data:
        rfecv_url = supabase.storage.from_('Visualisations').get_public_url(rfecv_result.data[0]['file_path'])
        st.image(rfecv_url)
    else:
        st.write("RFECV plot not available for this model.")

    st.subheader("Feature Importance")
    importance_result = supabase.table('visualizations').select('*').eq('model_id', result['id']).eq('vis_type', 'importance').execute()
    if importance_result.data:
        importance_url = supabase.storage.from_('Visualisations').get_public_url(importance_result.data[0]['file_path'])
        st.image(importance_url)
    else:
        st.write("Feature importance plot not available for this model.")

    # Venn diagram for feature subsets
    st.subheader("Feature Subset Intersection")
    venn_result = supabase.table('visualizations').select('*').eq('model_id', result['id']).eq('vis_type', 'venn').execute()
    if venn_result.data:
        venn_url = supabase.storage.from_('Visualisations').get_public_url(venn_result.data[0]['file_path'])
        st.image(venn_url)
    else:
        st.write("Venn diagram not available for the selected number of feature subsets.")

def display_xgboost_visuals(supabase, result):
    # Feature importance
    st.subheader("Feature Importance")
    importance_result = supabase.table('xgboost_visualizations').select('*').eq('model_id', result['id']).eq('vis_type', 'importance').execute()
    if importance_result.data:
        importance_url = supabase.storage.from_('Tree_Visuals').get_public_url(importance_result.data[0]['file_path'])
        st.image(importance_url)
    else:
        st.write("Feature importance plot not available for this model.")

    visualisation_table = supabase.table('xgboost_visualizations').select('*').eq('model_id', result['id']).execute()
    if visualisation_table.data:
        vis_table = pd.DataFrame(visualisation_table.data)

        st.subheader("Cross Validation Results")
        cv_url = supabase.storage.from_('Tree_Visuals').get_public_url(vis_table[vis_table['vis_type'] == 'rfecv']['file_path'].values[0])
        st.image(cv_url)

        st.subheader("Learning Curve")
        learning_url = supabase.storage.from_('Tree_Visuals').get_public_url(vis_table[vis_table['vis_type'] == 'learning_curve']['file_path'].values[0])
        st.image(learning_url)

        st.subheader("Parity Plot")
        parity_url = supabase.storage.from_('Tree_Visuals').get_public_url(vis_table[vis_table['vis_type'] == 'parity']['file_path'].values[0])
        st.image(parity_url)

        st.subheader("Feature Importance")
        importance_url = supabase.storage.from_('Tree_Visuals').get_public_url(vis_table[vis_table['vis_type'] == 'feature_importance']['file_path'].values[0])
        st.image(importance_url)

        st.subheader("SHAP Plots")
        st.write("Network Plot")
        network_url = supabase.storage.from_('Tree_Visuals').get_public_url(vis_table[vis_table['vis_type'] == 'network']['file_path'].values[0])
        st.image(network_url)

        st.write("N_Sii Plot")
        n_sii_url = supabase.storage.from_('Tree_Visuals').get_public_url(vis_table[vis_table['vis_type'] == 'n_sii']['file_path'].values[0])
        st.image(n_sii_url)

        st.write("Force Plot")
        force_url = supabase.storage.from_('Tree_Visuals').get_public_url(vis_table[vis_table['vis_type'] == 'force_SV']['file_path'].values[0])
        st.image(force_url)

        st.write("Force Interaction Plot")
        force_int_url = supabase.storage.from_('Tree_Visuals').get_public_url(vis_table[vis_table['vis_type'] == 'force_n_SII']['file_path'].values[0])
        st.image(force_int_url)

        st.write("Waterfall Plot")
        waterfall_url = supabase.storage.from_('Tree_Visuals').get_public_url(vis_table[vis_table['vis_type'] == 'waterfall_SV']['file_path'].values[0])
        st.image(waterfall_url)

        st.write("Waterfall Interaction Plot")
        waterfall_int_url = supabase.storage.from_('Tree_Visuals').get_public_url(vis_table[vis_table['vis_type'] == 'waterfall_n_sii']['file_path'].values[0])
        st.image(waterfall_int_url)
    else:
        st.write("No visualizations available for this model.")      

def display_model_details(supabase, result, model_type):
    st.header(f"{model_type} Results")
    col1, col2, col3 = st.columns(3)

    col1.metric("R-squared (test)", f"{result['r_squared']:.4f}", delta=None)
    col2.metric("RMSE (test)", f"{result['rmse']:.4f}")
    col3.metric("MAE (test)", f"{result['mae']:.4f}")

    # Display visualizations
    if model_type == "Linear Regression":
        display_linear_regression_visuals(supabase, result)
    else:
        display_xgboost_visuals(supabase, result)
    # Display selected features
    st.subheader("Selected Features")
    if model_type == "Linear Regression":
        selected_features_result = supabase.table('selected_features').select('*').eq('model_id', result['id']).execute()
    else:
        selected_features_result = supabase.table('xgboost_selected_features').select('*').eq('model_id', result['id']).execute()

    if selected_features_result.data:
        st.write(", ".join([f['feature_name'] for f in selected_features_result.data]))
    else:
        st.write("No selected features information available for this model.")

    # Display feature importance
    st.subheader("Feature Importance Details")
    if model_type == "Linear Regression":
        feature_importance_result = supabase.table('feature_importance').select('*').eq('model_id', result['id']).execute()
    else:
        feature_importance_result = supabase.table('xgboost_feature_importance').select('*').eq('model_id', result['id']).execute()

    if feature_importance_result.data:
        importance_df = pd.DataFrame(feature_importance_result.data)
        importance_df = importance_df.sort_values('importance', ascending=False)
        st.dataframe(importance_df)
    else:
        st.write("No feature importance details available for this model.")

    # Display coefficient information (for Linear Regression) or hyperparameters (for XGBoost)
    if model_type == "Linear Regression":
        st.subheader("Coefficient Information")
        coefficient_info_result = supabase.table('coefficient_info').select('*').eq('model_id', result['id']).execute()
        if coefficient_info_result.data:
            coef_df = pd.DataFrame(coefficient_info_result.data)
            coef_df = coef_df.sort_values('p_value')
            st.dataframe(coef_df)
        else:
            st.write("No coefficient information available for this model.")
    else:
        st.subheader("Hyperparameters")
        hyperparameters_result = supabase.table('xgboost_hyperparameters').select('*').eq('model_id', result['id']).execute()
        if hyperparameters_result.data:
            hyperparam_df = pd.DataFrame(hyperparameters_result.data)
            st.dataframe(hyperparam_df)
        else:
            st.write("No hyperparameter information available for this model.")  


st.title("Regression Analysis Dashboard")
supabase = init_connection()

tab1, tab2 = st.tabs(["Linear Regression", "XGBoost"])

with tab1:
    # st.header("Linear Regression")
    model_type = "Linear Regression"
    col1, col2 = st.columns(2)
    with col1:
        selected_features = st.multiselect("Select feature subsets:", options=list(feature_dictionary.keys()), default=list(feature_dictionary.keys())[0], key="linear_regression_features")
    with st.spinner("Fetching response variables..."):
        try:
            if st.session_state.regression_table is None:
                st.session_state.regression_table = get_data('regression_models')
            response_vars = st.session_state.regression_table
            response_var_options = [r['response_variable'] for r in response_vars if set(r['feature_subset']) == set(selected_features)]
        except Exception as e:
            st.error(f"Error fetching response variables for Linear Regression: {e}")
    if not response_var_options:
        response_var_options = ["No available targets for selected features"]
    with col2:
        response_var = st.selectbox("Select target variable:", response_var_options, key="target_select_lr")
    results = st.session_state.regression_table
    # results = get_data('regression_models')
    filtered_results = [r for r in results if set(r['feature_subset']) == set(selected_features) and r['response_variable'] == response_var]
    if not filtered_results:
        st.warning(f"No results found for the selected combination of features and target variable for {model_type}.")
    else:    
        result = filtered_results[0]  # Assume one result per combination
        display_model_details(supabase, result, model_type)
with tab2:
    # st.header("XGBoost")
    model_type = "XGBoost"
    col1, col2 = st.columns(2)
    with col1:
        selected_features = st.multiselect("Select feature subsets:", options=list(feature_dictionary.keys()), default=list(feature_dictionary.keys())[0], key="xgboost_features")
    with st.spinner("Fetching response variables..."):
        try:
            if st.session_state.xgboost_table is None:
                st.session_state.xgboost_table = get_data('xgboost_models')
            response_vars = st.session_state.xgboost_table
            response_var_options = [r['response_variable'] for r in response_vars if set(r['feature_subset']) == set(selected_features)]
        except Exception as e:
            st.error(f"Error fetching response variables for XGBoost: {e}")
    if not response_var_options:
        response_var_options = ["No available targets for selected features"]
    with col2:
        response_var = st.selectbox("Select target variable:", response_var_options, key="target_select_xgb")
    results = st.session_state.xgboost_table
    # results = get_data('xgboost_models')
    filtered_results = [r for r in results if set(r['feature_subset']) == set(selected_features) and r['response_variable'] == response_var]
    if not filtered_results:
        st.warning(f"No results found for the selected combination of features and target variable for {model_type}.")
    else:    
        result = filtered_results[0]  # Assume one result per combination
        display_model_details(supabase, result, model_type)
    

# Sidebar for user inputs
# st.sidebar.header("Model Selection")
# model_type = st.sidebar.radio("Select model type:", ["Linear Regression", "XGBoost"])

# st.sidebar.header("Feature Selection")
# selected_features = st.sidebar.multiselect(
#     "Select feature subsets:",
#     options=list(feature_dictionary.keys()),
#     default=list(feature_dictionary.keys())[0]
# )

# response_var_options = []
# if model_type == "Linear Regression":
#     with st.spinner("Fetching response variables..."):
#         try:
#             if st.session_state.regression_table is None:
#                 st.session_state.regression_table = get_data('regression_models')
#             # response_vars = get_data('regression_models')
#             response_vars = st.session_state.regression_table
#             response_var_options = [r['response_variable'] for r in response_vars if set(r['feature_subset']) == set(selected_features)]
#         except Exception as e:
#             st.error(f"Error fetching response variables for Linear Regression: {e}")
        
# else:
#     with st.spinner("Fetching response variables..."):
#         try:
#             if st.session_state.xgboost_table is None:
#                 st.session_state.xgboost_table = get_data('xgboost_models')
#             response_vars = st.session_state.xgboost_table
#             # response_vars = get_data('xgboost_models')
#             response_var_options = [r['response_variable'] for r in response_vars if set(r['feature_subset']) == set(selected_features)]
#         except Exception as e:
#             st.error(f"Error fetching response variables for XGBoost: {e}")
# if not response_var_options:
#     response_var_options = ["No available targets for selected features"]

# response_var = st.sidebar.selectbox("Select target variable:", response_var_options, key="target_select")

# Fetch results from database
# if model_type == "Linear Regression":
#     results = st.session_state.regression_table
#     # results = get_data('regression_models')
#     filtered_results = [r for r in results if set(r['feature_subset']) == set(selected_features) and r['response_variable'] == response_var]
# else:  # XGBoost
#     results = st.session_state.xgboost_table
#     # results = get_data('xgboost_models')
#     filtered_results = [r for r in results if set(r['feature_subset']) == set(selected_features) and r['response_variable'] == response_var]

# if not filtered_results:
#     st.warning(f"No results found for the selected combination of features and target variable for {model_type}.")
# else:    
#     result = filtered_results[0]  # Assume one result per combination

# Display results
    

