import pandas as pd
import numpy as np
import sys
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# Cargar datos desde archivo local con muestreo estratificado
def cargar_datos_con_muestreo(filepath):
    df = pd.read_csv(filepath)

    # Asegurar tipos correctos
    columnas_categoricas = ['cole_area_ubicacion', 'cole_bilingue', 'cole_calendario',
                            'cole_caracter', 'cole_genero', 'cole_jornada', 'cole_naturaleza',
                            'estu_genero', 'estu_tipodocumento', 'fami_educacionmadre',
                            'fami_educacionpadre', 'fami_estratovivienda', 'nivel_recursos']
    for col in columnas_categoricas:
        if col in df.columns:
            df[col] = df[col].astype(str)

    columnas_numericas = ['punt_ingles', 'punt_matematicas', 'punt_sociales_ciudadanas',
                          'punt_c_naturales', 'punt_lectura_critica', 'punt_global', 'indice_recursos']
    for col in columnas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Realizar muestreo estratificado por periodo
    df_2019 = df[df['periodo'] == '2019'].sample(n=250000, random_state=42)
    df_2022 = df[df['periodo'] == '2022'].sample(n=250000, random_state=42)
    df_muestra = pd.concat([df_2019, df_2022], ignore_index=True)

    return df_muestra


# Crear modelos predictivos por área
def crear_modelos_por_area(df):
    variables_categoricas = ['cole_area_ubicacion', 'cole_bilingue', 'cole_calendario',
                             'cole_naturaleza', 'nivel_recursos']

    areas = {
        'Lectura Crítica': 'punt_lectura_critica',
        'Matemáticas': 'punt_matematicas',
        'Ciencias Naturales': 'punt_c_naturales',
        'Sociales y Ciudadanas': 'punt_sociales_ciudadanas',
        'Inglés': 'punt_ingles',
        'Global': 'punt_global'
    }

    modelos = {}
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), variables_categoricas)
        ])

    for nombre_area, variable in areas.items():
        if variable not in df.columns:
            continue

        X = df[variables_categoricas].dropna()
        y = df[variable].dropna()

        # Alinear índices
        X = X.loc[y.index]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        modelos[nombre_area] = {
            'modelo': modelo,
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }

    return modelos


# Crear interfaz interactiva
def crear_interfaz_streamlit(modelos, df):
    st.title("Predictor de Resultados Saber 11 por Área")
    st.sidebar.title("Filtros de Configuración")

    # Filtros
    area_ubicacion = st.sidebar.selectbox("Área de Ubicación", ['URBANO', 'RURAL'])
    bilingue = st.sidebar.selectbox("¿Es Bilingüe?", ['N', 'S'])
    calendario = st.sidebar.selectbox("Calendario", ['A', 'B', 'OTRO'])
    naturaleza = st.sidebar.selectbox("Naturaleza", ['OFICIAL', 'NO OFICIAL'])
    nivel_recursos = st.sidebar.selectbox("Nivel de Recursos", df['nivel_recursos'].unique())

    if st.sidebar.button("Predecir"):
        datos_prediccion = pd.DataFrame({
            'cole_area_ubicacion': [area_ubicacion],
            'cole_bilingue': [bilingue],
            'cole_calendario': [calendario],
            'cole_naturaleza': [naturaleza],
            'nivel_recursos': [nivel_recursos]
        })

        predicciones = []
        for nombre_area, modelo_info in modelos.items():
            prediccion = modelo_info['modelo'].predict(datos_prediccion)[0]
            predicciones.append({
                'Área': nombre_area,
                'Puntaje Predicho': prediccion
            })

        resultados_df = pd.DataFrame(predicciones)

        # Mostrar puntaje global como indicador
        puntaje_global = resultados_df[resultados_df['Área'] == 'Global']['Puntaje Predicho'].values[0]
        st.markdown(f"### Puntaje Global Estimado: **{puntaje_global:.1f}**")

        # Radar chart para comparar áreas
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=resultados_df[resultados_df['Área'] != 'Global']['Puntaje Predicho'],
            theta=resultados_df[resultados_df['Área'] != 'Global']['Área'],
            fill='toself',
            name='Puntajes por Área'
        ))
        fig_radar.update_layout(title="Comparación de Puntajes por Área (Radar Chart)")
        st.plotly_chart(fig_radar)

        # Visualizaciones de áreas
        st.subheader("Distribución de Puntajes por Área")
        fig = px.bar(resultados_df, x='Área', y='Puntaje Predicho', title="Resultados Predichos por Área")
        st.plotly_chart(fig)

    # Insights con visualizaciones adicionales
    st.subheader("Exploración de Datos por Área")
    seleccion_area = st.selectbox("Selecciona el área a analizar:", list(modelos.keys()))
    if seleccion_area:
        columna_area = f"punt_{seleccion_area.lower().replace(' ', '_')}"
        if columna_area in df.columns:
            fig_box = px.box(df, y=columna_area, color='cole_naturaleza',
                             title=f"Distribución de {seleccion_area} por Naturaleza del Colegio")
            st.plotly_chart(fig_box)

            fig_scatter = px.scatter(df, x='nivel_recursos',
                                     y=columna_area,
                                     color='cole_area_ubicacion',
                                     title=f"Relación entre Nivel de Recursos y {seleccion_area}")
            st.plotly_chart(fig_scatter)


# Ejecución principal
if __name__ == "__main__":
    ruta_datos = "datos_completos_final.csv"  # Ruta del archivo local
    datos = cargar_datos_con_muestreo(ruta_datos)
    modelos = crear_modelos_por_area(datos)
    crear_interfaz_streamlit(modelos, datos)
