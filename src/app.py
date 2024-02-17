import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from data_cleaner import get_dataframe

# Certifique-se de ter as stopwords do nltk baixadas
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    # Limpeza de dados
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])

    # Tokenização
    tokens = word_tokenize(text)

    # Remoção de stopwords
    stop_words = set(stopwords.words('portuguese'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming (ou lematização, escolha uma)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Junte os tokens de volta em uma string
    processed_text = ' '.join(tokens)

    return processed_text

def recomendar_prato(input_ingredientes, base_de_dados, n_recomendacoes=3):
    # Aplicação da função de pré-processamento aos ingredientes de entrada
    input_ingredientes_preprocessados = preprocess_text(input_ingredientes)

    # Criação da matriz TF/IDF com parâmetros ajustados
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=0.1, ngram_range=(1, 2))
    matriz_tfidf = vectorizer.fit_transform(base_de_dados['Ingredientes'] + ' ' + input_ingredientes_preprocessados)

    # Cálculo da similaridade de cosseno
    similaridades = cosine_similarity(matriz_tfidf[:-1], matriz_tfidf[-1])

    # Índices dos pratos mais similares (top N)
    indices_pratos_similares = similaridades.argsort(axis=0)[-n_recomendacoes:]

    # Pratos recomendados
    pratos_recomendados = base_de_dados.iloc[indices_pratos_similares.flatten(), base_de_dados.columns.get_loc('Nome')]

    return pratos_recomendados

# Seu código original
inputs_gerados = [
    "Manteiga, Ovos e Frango",
    "Banana, Aveia e Mel",
    "Queijo, Tomate e Manjericão",
    "Arroz, Feijão e Carne",
    "Chocolate, Morango e Creme",
    "Abacate, Limão e Mel",
    "Cogumelos, Alho e Azeite",
    "Espinafre, Ovos e Queijo",
    "Peito de Frango, Batata Doce e Brócolis",
    "Pasta de Amendoim, Banana e Granola",
    "Salmão, Aspargos e Limão",
    "Maçã, Canela e Aveia",
    "Açaí, Banana e Granola",
    "Abóbora, Gengibre e Canela",
    "Iogurte, Morango e Chia",
    "Tomate, Manjericão e Mozzarella",
    "Cenoura, Alho-Poró e Batata",
    "Quinoa, Legumes e Frango Grelhado",
    "Kiwi, Melancia e Uva",
    "Peru, Abacaxi e Molho de Mostarda"
]

dataframe = get_dataframe()  # Use o DataFrame original ou o que você estiver usando

for input_ingrediente in inputs_gerados:
    pratos_recomendados = recomendar_prato(input_ingrediente, dataframe)
    print("Pratos Recomendados:")
    print(pratos_recomendados)

