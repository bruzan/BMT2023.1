# Importando bibliotecas
import pandas as pd
import ast
import numpy as np

# Funções para leitura do arquivo cfg
cfg = 'config/index.cfg'

def read_leia(cfg):
    try:
        with open(cfg, 'r') as file:
            for line in file:
                if 'LEIA=' in line:
                    result = line.split('LEIA=')[1]
                    return result.strip()
    except FileNotFoundError:
        print("File 'index.cfg' not found.")

def read_escreva(cfg):
    try:
        with open(cfg, 'r') as file:
            for line in file:
                if 'ESCREVA=' in line:
                    result = line.split('ESCREVA=')[1]
                    return result.strip()
    except FileNotFoundError:
        print("File 'index.cfg' not found.")

# Função que checa se a string é um número
def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

# Função que filtra as colunas do dataframe
def filtrar_colunas(df):
    for coluna in df.columns:
        if is_number(coluna):
            df = df.drop(coluna,axis=1)
        elif len(coluna)<2:
            df = df.drop(coluna,axis=1)
    return df

# Carregando os dados
data = pd.read_csv(read_leia(cfg),delimiter=';',header=None)
# Transformando as strings na segunda coluna em listas
for i in range(len(data[1])):
    data[1][i] = ast.literal_eval(data[1][i])
# Calculando as frequencias de cada palavra em cada documento
freqs = {}
for n in range(len(data[1])):
    my_dict = {i:data[1][n].count(i) for i in data[1][n]}
    freqs[data[0][n]] = my_dict

# Calculando o número de documentos
values = []
ndocs = 0

for i in freqs:
    if max(freqs[i].keys()) > ndocs:
        ndocs = max(freqs[i].keys())

# Calculando a frequencia de cada palavra em cada documento
words_num = []
for i in range(0,ndocs):
    words_num.append(0)
for i in freqs:
    for j in freqs[i]:
        words_num[j-1] +=  j

# Criando um novo dataframe, calculando o tf-idf para cada termo e colocando no dataframe
df = pd.DataFrame(columns = freqs.keys(), index = range(1,1240))

for i in freqs:
    for j in freqs[i]:
        df.iloc[j-1][i] = (freqs[i][j]/words_num[j-1])*np.log(( ndocs / len(freqs[i])))

df = df.fillna(0) # Substituindo os valores NaN por 0
filtered_df = filtrar_colunas(df)

# Salvando a matriz em arquivo

filtered_df.to_csv(read_escreva(cfg), index = True, sep = ';')


