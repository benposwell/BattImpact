import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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
                                      'Company concentration', '(Non) compliance with social standards'],
    'Economic Feature': ['Price (latest, 1998)']
                                    
}

st.title("Understand a Battery")

st.write("This page allows you to understand the multidimensional impact of a specific battery. Simply pick a battery from the Materials Project [Cite] (MP) database, and see how it compares!")

@st.cache_data
def get_dataset():
    df = pd.read_csv('data/mp_total_encoded_normal.csv')
    df = df[df['energy_grav'] >= 0]

    working_ion_encoded = pd.get_dummies(df['working_ion'], prefix='working_ion')
    df = pd.concat([df, working_ion_encoded], axis=1)
    df = df.drop('working_ion', axis=1)
    df[working_ion_encoded.columns] = working_ion_encoded[working_ion_encoded.columns].astype(int)
    
    return df

data = get_dataset()

tab1, tab2 = st.tabs(["Single Battery Viewer", "Structural Comparison"])

with tab1:
    selected_battery = st.selectbox("Select a battery to understand", data["battery_id"].unique())
    feature_subset = st.selectbox("Select a feature subset to understand", feature_dictionary.keys())

    selected_features = feature_dictionary[feature_subset]
    stats = data[selected_features].agg(['min', 'median', 'mean', 'max', 'std'])
    q1 = data[selected_features].quantile(0.25)
    q3 = data[selected_features].quantile(0.75)

    stats = pd.concat([stats.loc['min'], q1, stats.loc['median'], stats.loc['mean'], q3, stats.loc['max'], stats.loc['std']], axis=1)
    stats.columns = ['min', 'q1', 'median', 'mean', 'q3', 'max', 'std']
    stats = stats.T

    battery_values = data[data['battery_id'] == selected_battery][selected_features].iloc[0]

    fig = go.Figure()
    for feature in selected_features:
        fig.add_trace(go.Box(
            x0=feature,
            y=data[feature],
            name=feature,
            boxpoints=False,
            marker_color='lightblue',
            line_color='darkblue'
        ))
        fig.add_trace(go.Scatter(
            x=[feature],
            y=[battery_values[feature]],
            mode='markers',
            name=f'{selected_battery} - {feature}',
            marker=dict(color='red', size=10, symbol='star')
        ))

    fig.update_layout(
        title=f"{feature_subset} Features for {selected_battery}",
        xaxis_title="Features",
        yaxis_title="Values",
        showlegend=False,
        height=600,
        width=800
    )

    st.plotly_chart(fig)

    st.write("Feature Statistics:")
    st.dataframe(stats.T)

    st.write(f"Values for {selected_battery}:")
    st.dataframe(battery_values.to_frame().T)
with tab2:
    elements = [col.split('_')[0] for col in data.columns if col.endswith('_formula_discharge')]
    # selected_element = st.selectbox("Select an element", elements)
    
    selected_elements = st.multiselect("Select elements to compare", elements)

    element_data = data[data[[f'{element}_formula_discharge' for element in selected_elements]].gt(0).all(axis=1)]

    feature_subset = st.selectbox("Select a feature subset", list(feature_dictionary.keys()), key="element_subset")
    selected_features = feature_dictionary[feature_subset]

    def calculate_stats(df):
        stats = df[selected_features].agg(['min', 'median', 'mean', 'max', 'std'])
        q1 = df[selected_features].quantile(0.25)
        q3 = df[selected_features].quantile(0.75)

        stats = pd.concat([stats.loc['min'], q1, stats.loc['median'], stats.loc['mean'], q3, stats.loc['max'], stats.loc['std']], axis=1)
        stats.columns = ['min', 'q1', 'median', 'mean', 'q3', 'max', 'std']
        return stats.T

    all_stats = calculate_stats(element_data)
    element_stats = calculate_stats(element_data)

    fig = go.Figure()

    for feature in selected_features:
        # Box plot for all data
        fig.add_trace(go.Box(
            x0=feature,
            y=data[feature],
            name=f"All - {feature}",
            boxpoints=False,
            showlegend=False,
            marker_color='lightblue',
            line_color='darkblue'
        ))
        
        # Box plot for element-specific data
        fig.add_trace(go.Box(
            x0=feature,
            y=element_data[feature],
            name=f"{selected_elements} - {feature}",
            boxpoints=False,
            showlegend=False,
            marker_color='lightgreen',
            line_color='darkgreen'
        ))

    fig.update_layout(
        title=f"{feature_subset} Features for {selected_elements}-containing Batteries vs All Batteries",
        xaxis_title="Features",
        yaxis_title="Values",
        showlegend=False,
        height=600,
        width=800
    )

    # Add legend to denote color meanings
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color='darkblue'),
        name='All Batteries'
    ))
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color='darkgreen'),
        name=f'{selected_elements}-containing Batteries'
    ))

    fig.update_layout(showlegend=True)

    st.plotly_chart(fig)

    # Display statistics tables
    col1, col2 = st.columns(2)
    with col1:
        st.write("All Batteries Statistics:")
        st.dataframe(all_stats)
    with col2:
        st.write(f"{selected_elements}-containing Batteries Statistics:")
        st.dataframe(element_stats)

    # Display number of batteries in each group
    st.write(f"Number of all batteries: {len(data)}")
    st.write(f"Number of {selected_elements}-containing batteries: {len(element_data)}")



