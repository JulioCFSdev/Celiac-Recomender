import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_dataframe():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_directory, '..', 'dataset', 'Delicias da Debora - Cafe da manha  e Almoço - Cafe da Manha.csv')
    df = pd.read_csv(csv_path)
    
    return df

def exploratory_analysis(df):
    # Examinar as primeiras linhas do DataFrame
    print(df.head())

    # Exibe as colunas do DataFrame
    print(df.columns)

    # Verificar tipos de dados, estatísticas descritivas e identificar valores ausentes
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

    # Histogramas
    df.hist(figsize=(10, 8))
    plt.show()

    # Boxplots
    # Correção para seleção direta das colunas
    sns.boxplot(data=df[['Calorias (kcal)', 'Carboidratos (g)', 'Proteínas (g)', 'Gordura (g)']])
    plt.show()

    # Matriz de correlação
    correlation_matrix = df[['Calorias (kcal)', 'Carboidratos (g)', 'Proteínas (g)', 'Gordura (g)']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()

    # Identificar outliers e tratar, por exemplo, substituir por valores médios
    outliers = df[['Calorias (kcal)', 'Carboidratos (g)', 'Proteínas (g)', 'Gordura (g)']].apply(lambda x: (x - x.mean()).abs() > 2 * x.std())

    # Contagem de palavras frequentes nos Ingredientes
    ingredientes_frequentes = df['Ingredientes'].str.split(expand=True).stack().value_counts()
    print(ingredientes_frequentes.head())
