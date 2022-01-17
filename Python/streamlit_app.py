import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout='wide')


def plot_row(fig, df_json):
    df_json["main"] = df_json["main"].astype(str)
    df_json["parameters"] = df_json["parameters"].astype(str)
    col1, col2, col3 = st.columns([5, 2, 2])
    with col1:
        st.plotly_chart(fig)
    with col2:
        st.table(df_json["main"][df_json.main != "nan"])
    with col3:
        st.table(df_json["clustering_stats"][:8])


def create_row(name):
    df = pd.read_csv(f"../results/OUT{name}")
    df["cluster_id"] = df["cluster_id"].astype(str)
    fig = px.scatter(df, x="d0", y="d1", color="cluster_id", hover_data=["id", "is_core", "distance_calculations"])
    df_json = pd.read_json(f"../results/STAT{name}")

    st.title(f"{df_json['main']['input_filename'].split('/')[-1].split('.')[0]}")
    if df_json["parameters"]["algorithm"] == "DBSCAN":
        st.title(f"DBSCAN, eps={df_json['parameters']['Eps']}, minPts={df_json['parameters']['minPts']}")
    else:
        st.title(f"DBSCRN, k={df_json['parameters']['k']}")
    plot_row(fig, df_json)

st.title('Clustering')
create_row("_Opt-DBSCAN_example_D2_R12_m4_e2.000000_rMin.csv")

create_row("_Opt-DBSCRN_example_D2_R12_k3_rMin.csv")

create_row("_Opt-DBSCRN_complex9_D2_R3031_k6_rMin.csv")

create_row("_Opt-DBSCRN_complex9_D2_R3031_k5_rMin.csv")

create_row("_Opt-DBSCRN_complex9_D2_R3031_k4_rMin.csv")

create_row("_Opt-DBSCRN_cluto-t7-10k_D2_R10000_k5_rMin.csv")

create_row("_Opt-DBSCRN_cluto-t7-10k_D2_R10000_k4_rMin.csv")
