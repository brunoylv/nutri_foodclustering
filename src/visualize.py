# src/visualize.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def pca_scatter(df, pca_cols=("pca1", "pca2")):
    """
    Retorna um scatter plotly com pca1 x pca2 colorido por cluster.
    """
    for c in pca_cols:
        if c not in df.columns:
            raise ValueError(f"Coluna PCA ausente: {c}")
    if 'cluster' not in df.columns:
        raise ValueError("Coluna 'cluster' ausente no dataframe.")

    fig = px.scatter(df,
                     x=pca_cols[0],
                     y=pca_cols[1],
                     color=df['cluster'].astype(str),
                     hover_data=['food_name', 'category', 'nutri_score'])
    fig.update_layout(title="PCA: Clusters", legend_title="Cluster")
    return fig

def cluster_bar(df, feature='nutri_score'):
    """
    Retorna um bar chart plotly com média do feature por cluster.
    """
    if 'cluster' not in df.columns:
        raise ValueError("Coluna 'cluster' ausente no dataframe.")
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' ausente no dataframe.")

    agg = df.groupby('cluster')[feature].mean().reset_index()
    fig = px.bar(agg, x='cluster', y=feature, labels={'cluster':'Cluster', feature: f'Média {feature}'})
    fig.update_layout(title=f"Média de {feature} por cluster")
    return fig

def hist_nutriscore(df, feature='nutri_score'):
    """
    Histograma do NutriScore (ou outra feature).
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' ausente no dataframe.")
    fig = px.histogram(df, x=feature, nbins=20)
    fig.update_layout(title=f"Distribuição de {feature}")
    return fig

def radar_for_food(df, food_name, features=None):
    """
    Gera gráfico radar comparando o alimento com a média do cluster.
    features: lista de colunas numéricas a comparar.
    Retorna figura plotly.graph_objects.
    """
    if features is None:
        features = ["calories", "protein", "carbs", "fat", "iron", "vitamin_c"]

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Features ausentes para radar: {missing}")

    row = df[df['food_name'] == food_name]
    if row.empty:
        raise ValueError(f"Alimento '{food_name}' não encontrado no dataframe.")

    row = row.iloc[0]
    cluster = row['cluster']
    avg = df[df['cluster'] == cluster][features].mean()

    vals_food = [float(row[f]) for f in features]
    vals_cluster = [float(avg[f]) for f in features]

    categories = features + [features[0]]
    vals_food = vals_food + [vals_food[0]]
    vals_cluster = vals_cluster + [vals_cluster[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals_cluster, theta=categories, fill='toself', name=f'Cluster {cluster} (média)'))
    fig.add_trace(go.Scatterpolar(r=vals_food, theta=categories, fill='toself', name=food_name))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True,
                      title=f'Perfil nutricional: {food_name} vs média do cluster')
    return fig