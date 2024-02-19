import streamlit as st
from data_controller import get_dataframe, process_tfidf_matrix, recommend_dishes

def calculate_daily_caloric_needs(sexo, peso, altura, idade, fator_atividade):
    # Fórmula de Harris-Benedict
    if sexo.lower() == "masculino":
        gasto_calorico_base = 88.362 + (13.397 * peso) + (4.799 * altura * 100) - (5.677 * idade)
    elif sexo.lower() == "feminino":
        gasto_calorico_base = 447.593 + (9.247 * peso) + (3.098 * altura * 100) - (4.330 * idade)
    else:
        st.error("Sexo inválido. Por favor, insira 'Masculino' ou 'Feminino'.")
        return None

    # Gasto Calórico Diário considerando o fator de atividade
    gasto_calorico_diario = gasto_calorico_base * fator_atividade

    return gasto_calorico_diario

def main():
    st.title("Recomendação de Pratos celíacos com base no gasto calórico")

    # Carregamento do DataFrame e processamento TF/IDF
    dataframe = get_dataframe()
    tfidf_vectorizer, tfidf_matrix = process_tfidf_matrix(dataframe)

    # Interface Streamlit com navegação lateral
    st.sidebar.title("Dados do Usuário")
    st.sidebar.markdown("Insira seus dados para calcular o gasto calórico diário.")
    
    sexo = st.sidebar.radio("Sexo:", ["Masculino", "Feminino"])
    peso = st.sidebar.number_input("Peso (kg):", min_value=0.0)
    altura = st.sidebar.number_input("Altura (m):", min_value=0.0, format="%f")
    idade = st.sidebar.number_input("Idade:", min_value=0, max_value=150)
    fator_atividade = st.sidebar.selectbox("Fator de Atividade:", [1.2, 1.375, 1.55, 1.725, 1.9])

    # Botão para acionar o cálculo do gasto calórico diário
    gasto_calorico_diario = None
    if st.sidebar.button("Calcular Gasto Calórico Diário"):
        gasto_calorico_diario = calculate_daily_caloric_needs(sexo, peso, altura, idade, fator_atividade)
        if gasto_calorico_diario is not None:
            st.sidebar.success(f"Gasto calórico diário estimado: {gasto_calorico_diario:.2f} calorias.")

    return gasto_calorico_diario  # Retorna o gasto calórico diário calculado ou None

def show_recommended_dishes(gasto_calorico_diario):
    # Carregamento do DataFrame e processamento TF/IDF
    dataframe = get_dataframe()
    tfidf_vectorizer, tfidf_matrix = process_tfidf_matrix(dataframe)

    # Interface principal para recomendação de pratos
    st.title("Recomendação de Pratos")
    st.markdown("Insira os ingredientes desejados e receba as recomendações dos top 3 pratos.")

    # Campo de entrada de texto
    user_input = st.text_input("Digite os ingredientes desejados, separados por vírgula:")

    # Botão para acionar a recomendação de pratos, habilitado somente após o cálculo do gasto calórico diário
    if st.button("Recomendar Pratos"):
        # Lógica para recomendação de pratos
        user_ingredients = [ingrediente.strip() for ingrediente in user_input.split(',')]
        pratos_recomendados = recommend_dishes(','.join(user_ingredients), dataframe, tfidf_vectorizer, tfidf_matrix)

        # Exibição dos top 3 pratos recomendados com todos os atributos
        if not pratos_recomendados.empty:
            st.subheader("Top 3 Pratos Recomendados:")
            for _, prato in pratos_recomendados.iterrows():
                st.write(f"**Nome:** {prato['Nome']}")
                st.write(f"**Ingredientes:** {prato['Ingredientes']}")
                st.write(f"**Calorias (kcal):** {prato['Calorias (kcal)']}")
                st.write("---")
        else:
            st.warning("Nenhum prato encontrado com base nos ingredientes inseridos.")

if __name__ == "__main__":
    gasto_calorico_diario = main()
    show_recommended_dishes(gasto_calorico_diario)
