🥗 NutriCluster — NutriScore + Clusterização de Alimentos

Projeto desenvolvido para análise nutricional e clusterização de alimentos utilizando K-Means, PCA e um sistema de pontuação NutriScore, para inferir a qualidade nutricional de alimentos.
A aplicação inclui uma interface interativa construída em Streamlit para explorar grupos nutricionais de alimentos, visualizar os clusters gerados, aplicar filtros e ainda obter recomendações de alimentos semelhantes.

🚀 Funcionalidades do Projeto

✔️ Cálculo automático do NutriScore (0–100), um apontuação de alimentos com base em quão nutritivos são;
✔️ Normalização dos valores nutricionais;
✔️ Clusterização dos alimentos via K-Means;
✔️ Redução dimensional via PCA (2D);
✔️ Visualização interativa com Plotly;
✔️ Recomendações de alimentos substitutos/parecidos;
✔️ Filtros avançados por nutrientes.

📁 Estrutura do Projeto
nutri_foodclustering/
│
├── app/
│   └── streamlitapp.py               # Aplicação Streamlit principal
│
├── data/
│   └── food_nutrition_dataset.csv    # Dataset padrão
│
├── src/
│   ├── preprocess.py                 # Carregamento e normalização dos dados
│   ├── nutriscore.py                 # Cálculo do NutriScore
│   ├── clustering.py                 # K-Means e PCA
│   └── visualize.py                  # Gráficos e visualizações
│
├── requirements.txt
└── README.md

📦 Instalação
1️⃣ Clone o repositório
git clone https://github.com/SEU_USUARIO/nutri_foodclustering.git
cd nutri_foodclustering

2️⃣ Instale as dependências
pip install -r requirements.txt

▶️ Executando o Projeto
Execute o Streamlit: streamlit run app/streamlitapp.py

📊 Dataset

O projeto já inclui um dataset padrão:

data/food_nutrition_dataset.csv

🧠 Como funciona o NutriScore do projeto?

O NutriScore aqui não é o NutriScore oficial europeu, mas sim uma pontuação personalizada para fins acadêmicos, calculada com:

- Proteína

- Ferro

- Vitamina C

- Calorias

- Gordura

- Carboidratos

Depois, o valor é normalizado no valor entre 0–100.

O cálculo pode ser ajustado no código:

src/nutriscore.py

🔍 Lógica de Clusterização

A clusterização funciona em três etapas:

1️⃣ Pré-processamento

Remoção de NaNs

Normalização MinMaxScaler

2️⃣ K-Means

Aplicado sobre:
[calories, protein, carbs, fat, iron, vitamin_c]

3️⃣ PCA (2 componentes)

Permite visualizar os clusters em 2D no gráfico principal.

📈 Visualizações Inclusas

Scatter PCA (clusters)

Histograma do NutriScore

Barplot com médias por cluster

Radar plot individual por alimento

Tabela filtrável com o NutriScore

🧩 Recomendações por Similaridade

O sistema usa Nearest Neighbors (sklearn) para sugerir alimentos nutricionalmente parecidos com o item selecionado.

🙋 Autor

Bruno Antonio Leão Do Vale

Projeto para disciplina de Inteligência Artifical, do curso Bacharelado em Sistemas de Informação, UNESP.