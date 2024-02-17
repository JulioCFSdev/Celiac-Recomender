import streamlit as st
from backend import get_dataframe, process_tfidf_matrix, recommend_dishes

def main():
    # Carregamento do DataFrame e processamento TF/IDF
    dataframe = get_dataframe()
    tfidf_vectorizer, tfidf_matrix = process_tfidf_matrix(dataframe)

    # Interface Streamlit
    st.title("Recomendação de Pratos")
    st.markdown("Insira os ingredientes desejados e receba as recomendações dos top 3 pratos.")

    # Campo de entrada de texto
    user_input = st.text_input("Digite os ingredientes desejados, separados por vírgula:")

    # Botão para acionar a recomendação
    if st.button("Recomendar Pratos"):
        # Lógica para recomendação de pratos
        user_ingredients = [ingrediente.strip() for ingrediente in user_input.split(',')]
        pratos_recomendados = recommend_dishes(','.join(user_ingredients), dataframe, tfidf_vectorizer, tfidf_matrix)

        # Exibição dos top 3 pratos recomendados
        if not pratos_recomendados.empty:
            st.subheader("Top 3 Pratos Recomendados:")
            st.write(pratos_recomendados)
        else:
            st.warning("Nenhum prato encontrado com base nos ingredientes inseridos.")

if __name__ == "__main__":
    main()
