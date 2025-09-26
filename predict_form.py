# predict_form.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar pipeline entrenado
pipeline = joblib.load("models/decision_tree_pipeline.pkl")

def run_prediction_form(df):
    st.subheader("📝 Formulario de predicción")

    # Creamos un diccionario para los inputs
    user_input = {}
    
    # Variables categóricas
    cat_features = df.select_dtypes(include=["object"]).columns
    for col in cat_features:
        options = list(df[col].dropna().unique())
        default_value = options[0] if options else ""
        user_input[col] = st.selectbox(f"{col}", options, index=0)

    # Variables numéricas
    num_features = df.select_dtypes(include=[np.number]).columns
    for col in num_features:
        min_val = int(df[col].min())
        max_val = int(df[col].max())
        mean_val = int(df[col].mean())
        user_input[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

    # Convertimos a DataFrame
    new_data = pd.DataFrame([user_input])

    # Llenar columnas faltantes que tiene el pipeline
    for col in pipeline.feature_names_in_:
        if col not in new_data.columns:
            new_data[col] = 0  # o algún valor neutro

    # Botón para predecir
    if st.button("Predecir"):
        try:
            prediction = pipeline.predict(new_data)[0]
            proba = pipeline.predict_proba(new_data)[0][1]  # Probabilidad de "yes"

            # Convertimos la predicción a YES / NO
            if isinstance(prediction, str):
                prediction_label = prediction.upper()
            else:
                prediction_label = "YES" if prediction == 1 else "NO"

            st.success(f"Predicción: {prediction_label}")
            st.info(f"Probabilidad de aceptación: {proba*100:.2f}%")

        except Exception as e:
            st.error(f"Error al predecir: {e}")
