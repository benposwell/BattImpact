import streamlit as st
import pandas as pd
import plotly.express as px

target_option_dict = {
            'Structural Encoding': ['average_voltage', 'capacity_grav', 'energy_grav'],
             'Battery Properties': ['Price (latest, 1998)'],
             'Environmental Impact Features': ['Political Stability', 'Demand growth', 'Mining capacity', 
                                          'Concentration of reserves', 'Concentration of production', 
                                          'Trade barriers', 'Feasability of exploration projects', 
                                          'Price volatility', 'Occurence of co-production', 'Primary material use', 
                                          'Company concentration', '(Non) compliance with social standards', 'ADP (Kg)',
                                          'average_voltage', 'capacity_grav', 'energy_grav', 'Price (latest, 1998)', 'UK_Critical', 'US_Critical','EU_Critical'],
             'Socioeconomic Impact Features': ['CCH', 'ODP', 'HT', 'POF', 'PM', 'IR', 'CCE', 'TA', 
                                          'FE', 'TET', 'FET', 'MET', 'ALO', 'ULO', 'NLT', 'Human Health', 
                                          'Eco- systems', 'Criticality EI Score', 'US_Critical', 'EU_Critical', 'UK_Critical', 'average_voltage', 'capacity_grav', 'energy_grav'],
             ('Battery Properties', 'Environmental Impact Features'): ['Political Stability', 'Demand growth', 'Mining capacity', 
                                          'Concentration of reserves', 'Concentration of production', 
                                          'Trade barriers', 'Feasability of exploration projects', 
                                          'Price volatility', 'Occurence of co-production', 'Primary material use', 
                                          'Company concentration', '(Non) compliance with social standards', 'ADP (Kg)',
                                          'Price (latest, 1998)', 'UK_Critical', 'US_Critical','EU_Critical'],
             ('Battery Properties', 'Socioeconomic Impact Features'): ['CCH', 'ODP', 'HT', 'POF', 'PM', 'IR', 'CCE', 'TA', 
                                          'FE', 'TET', 'FET', 'MET', 'ALO', 'ULO', 'NLT', 'Human Health', 
                                          'Eco- systems', 'Criticality EI Score', 'US_Critical', 'EU_Critical', 'UK_Critical'],
             ('Environmental Impact Features', 'Socioeconomic Impact Features'): ['average_voltage', 'capacity_grav', 'energy_grav', 'UK_Critical', 'US_Critical','EU_Critical', 'Price (latest, 1998)'],
             ('Battery Properties', 'Environmental Impact Features', 'Socioeconomic Impact Features'): ['EU_Critical', 'UK_Critical', 'US_Critical', 'Price (latest, 1998)']
            }

# @st.cache_data
def load_data():
    results = pd.read_csv('data/tsne_results.csv')
    evaluations = pd.read_csv('data/tsne_evaluations.csv')
    return results, evaluations


st.title('t-SNE Visualisation Dashboard')
results, evaluations = load_data()

st.markdown("**Select Feature Subset and Target Variable**")
col1, col2 = st.columns(2)
with col1:
    feature_subsets = [col.split('_x')[0] for col in results.columns if col.endswith('_x')]
    selected_subset = st.selectbox('Select feature subset:', feature_subsets)
with col2:
    target_options = target_option_dict[selected_subset]
    selected_target = st.selectbox('Select target variable for color coding:', target_options)

# Display metrics for the selected subset
st.markdown(f'**Metrics for {selected_subset}**')
subset_metrics = evaluations[evaluations['method'] == selected_subset].iloc[0]
st.write(f"Trustworthiness: {subset_metrics['trustworthiness']:.4f}")
st.write(f"Continuity: {subset_metrics['continuity']:.4f}")
st.write(f"KL Divergence: {subset_metrics['kl_divergence']:.4f}")

# Prepare data for plotting
plot_data = results[[f'{selected_subset}_x', f'{selected_subset}_y', selected_target]]
plot_data.columns = ['x', 'y', 'target']

# Create plot
fig = px.scatter(plot_data, x='x', y='y', color='target',
                    color_continuous_scale='Spectral',
                    title=f't-SNE Visualization: {selected_subset}',
                    labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'target': selected_target},
                    hover_data='target')

# Update layout for better visibility
fig.update_layout(
    xaxis_title='t-SNE 1',
    yaxis_title='t-SNE 2',
    legend_title=selected_target
)

# Display the plot
st.plotly_chart(fig)


