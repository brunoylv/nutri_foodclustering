import streamlit as st
import pandas as pd
import sys
sys.path.append('.')
from src.preprocess import load_dataset, preprocess
from src.nutriscore import compute_raw_score, normalize_score
from src.clustering import run_kmeans, compute_pca
from src.visualize import pca_scatter, cluster_bar, hist_nutriscore, radar_for_food

st.set_page_config(layout='wide', page_title='NutriCluster v2', page_icon='🥗')

st.title('🥗 NutriCluster — NutriScore + Clusterização (v2)')
st.write("Checkpoint A: app iniciou")
st.markdown('Projeto: Clusterização + NutriScore. Use os filtros no painel esquerdo para explorar os dados.')

# Sidebar - upload and params
st.sidebar.header('Dados & Parâmetros')
uploaded = st.sidebar.file_uploader('Upload CSV (ou deixe vazio para usar data/food_nutrition_dataset.csv)', type=['csv'])

# carregar dataset (upload ou padrão)
try:
    if uploaded is None:
        df_raw = load_dataset('data/food_nutrition_dataset.csv')
    else:
        df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f'Erro ao carregar dataset: {e}')
    st.stop()

# normalize columns (sempre)
df_raw.columns = df_raw.columns.str.strip().str.lower()
st.write("Checkpoint B: dataset carregado", df_raw.shape)

# required columns
required_cols = ['food_name', 'category', 'calories', 'protein', 'carbs', 'fat', 'iron', 'vitamin_c']
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"Dataset não contém as colunas necessárias: {missing}. Verifique o CSV.")
    st.stop()

# ensure numeric for nutrient columns (safe coercion)
nutrient_cols = ['calories', 'protein', 'carbs', 'fat', 'iron', 'vitamin_c']
for c in nutrient_cols:
    df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')

# compute safe maxima for sliders (use 0.0 if all NaN)
safe_max = {}
for c in nutrient_cols:
    mx = df_raw[c].max(skipna=True)
    safe_max[c] = float(mx) if pd.notna(mx) else 0.0

st.sidebar.markdown('---')
st.sidebar.subheader('Seleção de K (k-means)')
k = st.sidebar.slider('Número de clusters (k)', min_value=2, max_value=8, value=4)

st.sidebar.subheader('Filtros de nutrientes (valores absolutos)')
cal_min, cal_max = st.sidebar.slider('Calories range', 0.0, safe_max['calories'], (0.0, safe_max['calories']))
prot_min, prot_max = st.sidebar.slider('Protein range', 0.0, safe_max['protein'], (0.0, safe_max['protein']))
carb_min, carb_max = st.sidebar.slider('Carbs range', 0.0, safe_max['carbs'], (0.0, safe_max['carbs']))
fat_min, fat_max = st.sidebar.slider('Fat range', 0.0, safe_max['fat'], (0.0, safe_max['fat']))
iron_min, iron_max = st.sidebar.slider('Iron range', 0.0, safe_max['iron'], (0.0, safe_max['iron']))
vitc_min, vitc_max = st.sidebar.slider('Vitamin C range', 0.0, safe_max['vitamin_c'], (0.0, safe_max['vitamin_c']))

st.sidebar.markdown('---')
top_n = st.sidebar.slider('Quantos itens mostrar na tabela', 5, 50, 15)
show_recs = st.sidebar.checkbox('Ativar recomendações por similaridade', value=True)
st.sidebar.markdown('---')
st.sidebar.write('Dataset deve ter colunas: food_name, category, calories, protein, carbs, fat, iron, vitamin_c')

# Preprocess and scoring
with st.spinner('Pré-processando...'):
    st.write("Checkpoint C: iniciando preprocess")
    try:
        df_scaled = preprocess(df_raw)
    except Exception as e:
        st.error(f"Erro em preprocess: {e}")
        st.stop()

    st.write("Checkpoint D: preprocess finalizado", df_scaled.shape)

    try:
        df_scored = compute_raw_score(df_scaled)
        df_scored = normalize_score(df_scored, feature='nutri_raw', out='nutri_score')
    except Exception as e:
        st.error(f"Erro ao computar NutriScore: {e}")
        st.stop()

    st.write("Checkpoint D2: nutri score calculado", df_scored.shape)

# Apply absolute filters on original-scale data (not scaled)
# ensure df_scored has original nutrient columns (convert if needed)
for c in nutrient_cols:
    if c not in df_scored.columns:
        # try to recover from df_raw
        if c in df_raw.columns:
            df_scored[c] = df_raw[c]
        else:
            df_scored[c] = 0.0

df_filtered = df_scored[
    (df_scored['calories']>=cal_min) & (df_scored['calories']<=cal_max) &
    (df_scored['protein']>=prot_min) & (df_scored['protein']<=prot_max) &
    (df_scored['carbs']>=carb_min) & (df_scored['carbs']<=carb_max) &
    (df_scored['fat']>=fat_min) & (df_scored['fat']<=fat_max) &
    (df_scored['iron']>=iron_min) & (df_scored['iron']<=iron_max) &
    (df_scored['vitamin_c']>=vitc_min) & (df_scored['vitamin_c']<=vitc_max)
].reset_index(drop=True)

st.write("Checkpoint E: filtros aplicados", df_filtered.shape)

if df_filtered.empty:
    st.warning('Nenhum alimento encontrado com esses filtros. Ajuste os valores.')
    st.stop()

st.write("Checkpoint F: iniciando kmeans")

# Clustering
with st.spinner('Executando k-means e PCA...'):
    try:
        df_clustered, kmodel = run_kmeans(df_filtered, k=k)
    except Exception as e:
        st.error(f"Erro ao rodar kmeans: {e}")
        st.stop()

    try:
        df_clustered, pca = compute_pca(df_clustered, n_components=2)
    except Exception as e:
        st.error(f"Erro ao rodar PCA: {e}")
        st.stop()

st.write("Checkpoint G: kmeans + pca finalizados", df_clustered.shape)

# Layout top
st.subheader('Tabela com NutriScore (filtrada)')
st.markdown('**Explicação**: NutriScore (0-100) — fórmula linear: +protein, +iron, +vitamin_c; -calories, -fat, -carbs. Ajuste os pesos em src/nutri_score.py.')
st.dataframe(df_clustered[['food_name','category','calories','protein','carbs','fat','iron','vitamin_c','nutri_score','cluster']].sort_values('nutri_score', ascending=False).head(top_n))

# Visualizations
col1, col2 = st.columns([2,1])
with col1:
    st.subheader('Clusters — PCA view')
    try:
        fig = pca_scatter(df_clustered)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro em pca_scatter: {e}")

    st.subheader('Distribuição do NutriScore')
    try:
        fig2 = hist_nutriscore(df_clustered)
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Erro em hist_nutriscore: {e}")

with col2:
    st.subheader('Estatísticas por cluster')
    try:
        fig3 = cluster_bar(df_clustered)
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.error(f"Erro em cluster_bar: {e}")

    st.markdown('**Médias (features normalizadas)**')
    st.dataframe(df_clustered.groupby('cluster')[['calories','protein','carbs','fat','iron','vitamin_c','nutri_score']].mean().round(3))

# Recommendations by similarity (optional)
if show_recs:
    st.subheader('Recomendações por similaridade')
    selected = st.selectbox('Escolha um alimento para ver substitutos', options=df_clustered['food_name'].tolist())
    if selected:
        from sklearn.neighbors import NearestNeighbors
        feats = ['calories','protein','carbs','fat','iron','vitamin_c']
        nn = NearestNeighbors(n_neighbors=10)
        nn.fit(df_clustered[feats].values)
        idx = df_clustered.index[df_clustered['food_name']==selected][0]
        dists, inds = nn.kneighbors([df_clustered.loc[idx,feats].values])
        recs = df_clustered.iloc[inds[0][1:6]][['food_name','category','nutri_score','calories','protein','fat','carbs']]
        st.table(recs.reset_index(drop=True))

# Export
st.sidebar.header('Exportar & Download')
if st.sidebar.button('Exportar CSV com nutri_score & cluster'):
    out = 'data/food_nutrition_scored_clustered.csv'
    df_clustered.to_csv(out, index=False)
    st.sidebar.success(f'Salvo em {out}')

st.markdown('---')
st.markdown('## Explicações detalhadas')
st.markdown('''\n**O que é NutriScore aqui?**\n\nÉ uma pontuação criada para o trabalho: combinação linear de nutrientes. Proteína, ferro e vitamina C contribuem positivamente; calorias, gordura e carboidratos contribuem negativamente. O resultado é normalizado para 0-100.\n\n**Por que clusterizar?**\n\nA clusterização (k-means) agrupa alimentos com perfis nutricionais semelhantes — útil para identificar grupos como alimentos ricos em proteína, alimentos calóricos ou ricos em vitaminas.\n\n**Interpretação dos resultados**\n\n- Observe as médias por cluster (painel direito) para entender o perfil de cada cluster.\n- Use o gráfico PCA para visualizar separações.\n- Use recomendações para encontrar substitutos com NutriScore mais alto.\n''')