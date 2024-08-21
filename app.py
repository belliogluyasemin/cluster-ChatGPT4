import os
import json
from google.cloud import secretmanager
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import openai
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="User Cluster",
    page_icon="icon.png",
)

current_dir = os.getcwd()

# Load Secret from GCP Secret Manager
def get_secret(secret_name, project_id, version_id='1'):
    client = secretmanager.SecretManagerServiceClient()
    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/{version_id}"
    response = client.access_secret_version(name=secret_path)
    return response.payload.data.decode('UTF-8')

# Set your GCP project ID and get OpenAI API key from Secret Manager
project_id = "psychic-root-424207-s9"
openai_api_key = get_secret("openai-api-key", project_id)

# Load Service Account credentials from Secret Manager and set the environment variable
service_account_info = get_secret("my-service-account-key", project_id)
service_account_path = "/tmp/service-account-key.json"

# Write the credentials to a file
with open(service_account_path, 'w') as f:
    f.write(service_account_info)

# Set the environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path

# Construct the full paths using the current directory
pickle_path = os.path.join(current_dir, "country_filtered_cluster_ward")
with open(pickle_path, 'rb') as file:
    country_filtered = pickle.load(file)
country_filtered.drop('cnt', axis=1, inplace=True)

pickle_path1 = os.path.join(current_dir, "xh_cluster_ward2")
with open(pickle_path1, 'rb') as file:
    x_h = pickle.load(file)
x_h.drop('cnt', axis=1, inplace=True)

# Streamlit page configuration
st.title('Hierarchical Clustering and Visualization')

# Add short explanation about Hierarchical Clustering
st.markdown("""
### What is Hierarchical Clustering?

Hierarchical clustering is a method of cluster analysis which seeks to build a hierarchy of clusters. It can be either agglomerative (bottom-up) or divisive (top-down). The agglomerative approach starts with each point as its own cluster and then repeatedly merges the closest pair of clusters, whereas the divisive approach starts with one cluster and recursively splits it.

This method is particularly useful when the goal is to understand the hierarchical relationships within the data, such as grouping customers based on purchasing behavior, as we are doing here.

#### Why Ward's Method?

In this project, we have chosen to use Ward's method for hierarchical clustering. Ward's method aims to minimize the variance within each cluster, leading to the most compact and homogeneous groups possible. This is particularly effective when the goal is to identify distinct, well-separated clusters.

To evaluate the quality of these clusters, metrics such as the Davies-Bouldin Score and the Calinski-Harabasz Score are used. The Davies-Bouldin Score measures the average similarity ratio of each cluster with its most similar cluster, where a lower score indicates better clustering. The Calinski-Harabasz Score, on the other hand, evaluates the ratio of the sum of between-cluster dispersion to within-cluster dispersion, with a higher score indicating a better-defined clustering structure.
""")

st.subheader("Dendrogram for Ward's Method")
image_path = os.path.join(current_dir, "icon.png")
st.image(image_path, caption="Cluster Analysis Icon", use_column_width=True)

# Dendrogram oluşturma ve plotlama
#fig, ax = plt.subplots(figsize=(10, 7))  
#dendrogram_ward = sch.dendrogram(sch.linkage(x_h, method='ward'), truncate_mode='level', p=4, ax=ax)
#st.pyplot(fig)

# Explanation of the dendrogram
st.markdown("""
As shown in the dendrogram above, the use of Ward's method has allowed us to separate the data into distinct clusters. The hierarchical clustering approach, combined with Ward's method, ensures that the clusters are as compact as possible while maintaining distinct boundaries between them.

The vertical distance between the clusters indicates the dissimilarity between them, and the large jumps seen here are a strong indicator that the data naturally separates into well-defined clusters. This visual evidence supports the validity of our clustering approach.
""")

# Sidebar for selecting column to visualize
st.sidebar.title("Controls")
numeric_cols = country_filtered.select_dtypes(include=['number']).columns
col_options = [col for col in numeric_cols if col != 'clusters_ward']

# Select "itemRevenue15" as the default column
if "itemRevenue15" in col_options:
    default_index = col_options.index("itemRevenue15")
    st.write('Select a feature from the left side to visualize the cluster bar graph below.')
else:
    default_index = 0  # If itemRevenue15 is not present, select the first column
    st.write('"itemRevenue15" is not available. The first numeric column will be used as the default.')

col_chosen = st.sidebar.radio("Choose a column to visualize with Plotly:", col_options, index=default_index)
st.subheader("Cluster Visuals with Plotly")

# Create the plot
fig = px.histogram(x_h, 
                   x='clusters_ward', 
                   y=col_chosen, 
                   histfunc='avg',
                   category_orders={'clusters_ward': [0, 1, 2]},
                   color='clusters_ward',  
                   color_discrete_sequence=px.colors.qualitative.Bold)

# Display the data table
st.write("Data Table")
st.dataframe(country_filtered.head(10), height=200, width=800)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# ChatGPT Support Section
st.subheader("Strategic Support - ChatGPT")

# Create DataFrame Agent
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0.7, model="gpt-4", openai_api_key=openai_api_key),
    country_filtered,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True,  # Considering security implications
)

# Get user question
user_question = st.text_input("Enter your question here:")

# Send the question to ChatGPT and display the response
if user_question:
    response = agent.run(user_question)
    st.write("Response:", response)
