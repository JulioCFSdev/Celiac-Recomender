import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from data_cleaner import get_dataframe

def process_tfidf_matrix(dataframe):
    ingredients_list = dataframe['Ingredientes'].tolist()
    tfidf_vectorizer_ingredients = TfidfVectorizer()
    tfidf_matrix_ingredients = tfidf_vectorizer_ingredients.fit_transform(ingredients_list)
    return tfidf_vectorizer_ingredients, tfidf_matrix_ingredients

def recommend_dishes(input_ingredientes, dataframe, tfidf_vectorizer, tfidf_matrix, n_recommendations=3, alpha=0.5):
    query_tfidf_ingredients = tfidf_vectorizer.transform([input_ingredientes])
    cosine_similarities_ingredients = linear_kernel(query_tfidf_ingredients, tfidf_matrix).flatten()
    combined_similarities = alpha * cosine_similarities_ingredients

    # Índices dos pratos mais similares (top N)
    indices_combined = combined_similarities.argsort()[-n_recommendations:]

    # Pratos recomendados
    pratos_recomendados = dataframe.iloc[indices_combined]

    return pratos_recomendados