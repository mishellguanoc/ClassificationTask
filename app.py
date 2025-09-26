import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from predict_form import run_prediction_form

# ===== 1. Configuración inicial =====
st.set_page_config(page_title="Bank Marketing EDA", layout="wide")
st.title("📊 Exploratory Data Analysis: Bank Marketing Dataset")

# ===== 2. Cargar dataset =====
data_path = "data/bank-full.csv"
df = pd.read_csv(data_path, sep=";")

st.write("### Vista previa del dataset")
st.dataframe(df.head())

# ===== 3. KPIs =====
st.subheader("🔑 KPIs principales")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total registros", len(df))
    st.metric("Aceptaron oferta (yes)", df["y"].value_counts().get("yes", 0))
with col2:
    st.metric("Rechazaron oferta (no)", df["y"].value_counts().get("no", 0))
    st.metric("Proporción aceptación (%)", round((df["y"].value_counts(normalize=True).get("yes", 0)) * 100, 2))

# ===== 4. Filtros dinámicos =====
st.sidebar.header("Filtros")

# Filtro por rango de edad
min_age, max_age = st.sidebar.slider("Rango de edad", int(df["age"].min()), int(df["age"].max()), (20, 60))
df_filtered = df[(df["age"] >= min_age) & (df["age"] <= max_age)]

# Filtro dinámico por variable categórica
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c != "y"]  # quitamos la target del filtro
filter_col = st.sidebar.selectbox("Selecciona variable categórica para filtrar:", cat_cols)

options = df[filter_col].unique().tolist()
selected_opts = st.sidebar.multiselect(f"Selecciona valores de {filter_col}:", options, default=options)

df_filtered = df_filtered[df_filtered[filter_col].isin(selected_opts)]

st.write(f"✅ Dataset filtrado: {df_filtered.shape[0]} registros")

# ===== 5. Visualizaciones dinámicas =====
st.subheader("📈 Visualizaciones Interactivas")

graph_type = st.selectbox(
    "Selecciona tipo de gráfico",
    ["Distribución categórica", "Histograma numérico", "Scatterplot", "Boxplot", "Heatmap de correlación"]
)

# Gráfico 1: Distribución categórica
if graph_type == "Distribución categórica":
    cat_cols = df_filtered.select_dtypes(include="object").columns.tolist()
    col = st.selectbox("Variable categórica:", cat_cols)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df_filtered, x=col, hue="y", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Gráfico 2: Histograma numérico
elif graph_type == "Histograma numérico":
    num_cols = df_filtered.select_dtypes(exclude="object").columns.tolist()
    col = st.selectbox("Variable numérica:", num_cols)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df_filtered[col], kde=True, bins=30, ax=ax)
    st.pyplot(fig)

# Gráfico 3: Scatterplot
elif graph_type == "Scatterplot":
    num_cols = df_filtered.select_dtypes(exclude="object").columns.tolist()
    x_var = st.selectbox("Eje X:", num_cols)
    y_var = st.selectbox("Eje Y:", num_cols)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df_filtered, x=x_var, y=y_var, hue="y", ax=ax, alpha=0.5)
    st.pyplot(fig)

# Gráfico 4: Boxplot
elif graph_type == "Boxplot":
    num_cols = df_filtered.select_dtypes(exclude="object").columns.tolist()
    y_var = st.selectbox("Variable numérica:", num_cols)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df_filtered, x="y", y=y_var, ax=ax)
    st.pyplot(fig)

# Gráfico 5: Heatmap
elif graph_type == "Heatmap de correlación":
    num_cols = df_filtered.select_dtypes(exclude="object").columns.tolist()
    corr = df_filtered[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ===== Cargar pipeline =====
pipeline = joblib.load("models/decision_tree_pipeline.pkl")


# ===== Llamar al formulario de predicción =====
run_prediction_form(df)