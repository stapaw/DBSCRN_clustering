import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px

PARAMETERS = "#parameters"

MAIN = "#main"

st.set_page_config(layout='wide')


def plot_row(fig, df_json):
    df_json[MAIN] = df_json[MAIN].astype(str)
    df_json[PARAMETERS] = df_json[PARAMETERS].astype(str)
    col1, col2, col3 = st.columns([5, 2, 2])
    with col1:
        st.plotly_chart(fig)
    with col2:
        st.table(df_json[MAIN][df_json[MAIN] != "nan"])
    with col3:
        st.table(df_json["clustering_stats"][:8])


def create_row(name):
    df = pd.read_csv(f"../results/OUT{name}")
    df["cluster_id"] = df["cluster_id"].astype(str)
    fig = px.scatter(df, x="d0", y="d1", color="cluster_id", hover_data=["id", "is_core", "distance_calculations"])
    df_json = pd.read_json(f"../results/STAT{name}")

    st.title(f"{df_json[MAIN]['input_file'].split('/')[-1].split('.')[0]}")
    if df_json[PARAMETERS]["algorithm"] == "DBSCAN":
        st.title(f"DBSCAN, eps={df_json[PARAMETERS]['eps']}, minPts={df_json[PARAMETERS]['minPts']}")
    else:
        st.title(f"DBSCRN, k={df_json[PARAMETERS]['k']}")
    plot_row(fig, df_json)

st.title('Clustering')
create_row("_Opt-DBSCRN_example_D2_R12_k3_minkowski2_refMin.csv")
create_row("_Opt-DBSCRN_complex9_D2_R3031_k5_minkowski2_refMin.csv")
create_row("_Opt-DBSCRN_complex9_D2_R3031_k6_minkowski2_refMin.csv")
create_row("_Opt-DBSCRN_complex9_D2_R3031_k7_minkowski2_refMin.csv")
create_row("_Opt-DBSCRN_complex9_D2_R3031_k8_minkowski2_refMin.csv")
create_row("_Opt-DBSCRN_complex9_D2_R3031_k9_minkowski2_refMin.csv")
create_row("_Opt-DBSCRN_complex9_D2_R3031_k10_minkowski2_refMin.csv")
create_row("_Opt-DBSCRN_complex9_D2_R3031_k20_minkowski2_refMin.csv")
create_row("_Opt-DBSCRN_complex9_D2_R3031_k25_minkowski2_refMin.csv")
