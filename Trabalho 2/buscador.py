# Importando bibliotecas
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
 

# Funções para leitura do arquivo cfg
cfg = 'config/busca.cfg'

def read_modelo(cfg):
    try:
        with open(cfg, 'r') as file:
            for line in file:
                if 'MODELO=' in line:
                    result = line.split('MODELO=')[1]
                    return result.strip()
    except FileNotFoundError:
        print("File 'busca.cfg' not found.")


def read_consulta(cfg):
    try:
        with open(cfg, 'r') as file:
            for line in file:
                if 'CONSULTAS=' in line:
                    result = line.split('CONSULTAS=')[1]
                    return result.strip()
    except FileNotFoundError:
        print("File 'busca.cfg' not found.")


def read_resultados(cfg):
    try:
        with open(cfg, 'r') as file:
            for line in file:
                if 'RESULTADOS=' in line:
                    result = line.split('RESULTADOS=')[1]
                    return result.strip()
    except FileNotFoundError:
        print("File 'busca.cfg' not found.")


# Escreve o dataframe no arquivo csv no formato especificado
def dataframe_to_csv(df, csv_filename):
    # Create a new dataframe with the desired structure
    new_df = pd.DataFrame()
    new_df[df.columns[0]] = df[df.columns[0]]
    new_df['List'] = df[df.columns[1:]].values.tolist()
    # Write the new dataframe to a CSV file
    new_df.to_csv(csv_filename, index=False,sep=';',header=False)


# Lendo as bases
querys = pd.read_csv(read_consulta(cfg),delimiter=';')
modelo = pd.read_csv(read_modelo(cfg),delimiter=';')
modelo = modelo.drop('Unnamed: 0',axis = 1)

# Gerando um lista com os textos das queries 
corpus = []
for i in range(len(querys['QueryText'])):
    corpus.append(querys['QueryText'][i])

# Transformando todas as letras em maiusculo
def uppercase(list):
    return [x.upper() for x in list]

corpus = uppercase(corpus)

# Gerando a matriz termo-documento para as queries 
vectorizer = TfidfVectorizer(lowercase=False, vocabulary = modelo.columns, use_idf = False, norm = None, binary = True)
querytfidf = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()
corpus_index = [n for n in corpus]
querymodel = pd.DataFrame(querytfidf.todense(), columns=feature_names, index = range(1,100))

# Calculate cosine similarity
similarity_matrix = cosine_similarity(querymodel.values, modelo.values)

# Convert similarity matrix to DataFrame
similarity_df = pd.DataFrame(similarity_matrix, columns=modelo.index, index=querymodel.index)
similarity_df['QueryNumber'] = similarity_df.index

# Fazendo um pivoteamento no dataframe
melted_df = similarity_df.melt(var_name = 'DocNumber', id_vars='QueryNumber', value_name='Distance')
melted_df['DocNumber'] = melted_df['DocNumber'].astype(int)

# Listando por numero de consulta e similaridade
melted_sorted = melted_df.sort_values(by=['QueryNumber', 'Distance'])

# Rankeando pela similaridade de cosseno
melted_sorted['Ranking'] = melted_sorted.groupby('QueryNumber')['Distance'].rank(method='first',ascending = False)

# Reorganizando em ordem crescente de numero de consulta, ranking, numero do documento e similaridade
sorted_data = melted_sorted.sort_values(['QueryNumber','Ranking','DocNumber','Distance'])
sorted_data = sorted_data[['QueryNumber','Ranking','DocNumber','Distance']]

# Escrevendo os resultado em um csv
dataframe_to_csv(sorted_data,read_resultados(cfg))



