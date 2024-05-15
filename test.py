import streamlit as st
import pandas as pd

# Crear un DataFrame de ejemplo llamado famous_top con 144 filas
famous_top = pd.DataFrame({
    'Book-Title': [f'Book {i}' for i in range(1, 145)],
    'Book-Author': [f'Author {i}' for i in range(1, 145)],
    'Book-Rating': [round((i % 5) + 1 + (i % 10) * 0.1, 1) for i in range(1, 145)]
})

# Verificar la estructura del DataFrame
st.write(famous_top.head())

# Configurar el número de columnas en el grid
num_columns = 8  # Número de columnas en el grid
rows = (len(famous_top) + num_columns - 1) // num_columns  # Calcular el número de filas

# Crear el grid de tarjetas
for i in range(rows):
    cols = st.columns(num_columns)  # Crear una fila con el número especificado de columnas
    for j in range(num_columns):
        index = i * num_columns + j
        if index < len(famous_top):  # Comprobar que el índice está dentro del rango del DataFrame
            with cols[j]:
                st.write(f"**Book-Title:** {famous_top.loc[index, 'Book-Title']}")
                st.write(f"**Book-Author:** {famous_top.loc[index, 'Book-Author']}")
                st.write(f"**Book-Rating:** {famous_top.loc[index, 'Book-Rating']}")
                st.markdown("---")  # Línea separadora entre tarjetas
