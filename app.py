import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    st.metric("Proporción aceptación (%)", round((df["y"].value_counts(normalize=True).get("yes", 0))*100, 2))

# ===== 4. Visualizaciones interactivas =====
st.subheader("📈 Visualizaciones Interactivas")

# Filtro dinámico
feature = st.selectbox("Selecciona una variable categórica:", df.select_dtypes(include=["object"]).columns)

# Barplot
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(data=df, x=feature, hue="y", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Scatter plot entre edad y duración de la llamada
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x="age", y="duration", hue="y", ax=ax2, alpha=0.5)
st.pyplot(fig2)

# Boxplot de balance vs aceptación
fig3, ax3 = plt.subplots()
sns.boxplot(data=df, x="y", y="balance", ax=ax3)
st.pyplot(fig3)
