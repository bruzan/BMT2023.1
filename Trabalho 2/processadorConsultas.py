# Importando as bibliotecas
import csv
import xml.etree.ElementTree as ET


# Definindo o arquivo de configuração
cfg = 'config/pc.cfg'

# Função que pega o arquivo na instrução leia
def read_leia(cfg):
    try:
        with open(cfg, 'r') as file:
            for line in file:
                if 'LEIA=' in line:
                    result = line.split('LEIA=')[1]
                    return result.strip()
    except FileNotFoundError:
        print("File 'pc.cfg' not found.")

# Função que pega o arquivo na instrução de consulta
def read_consulta(cfg):
    try:
        with open(cfg, 'r') as file:
            for line in file:
                if 'CONSULTAS=' in line:
                    result = line.split('CONSULTAS=')[1]
                    return result.strip()
    except FileNotFoundError:
        print("File 'pc.cfg' not found.")

# Função que pega o arquivo na instrução de esperados
def read_esperado(cfg):
    try:
        with open(cfg, 'r') as file:
            for line in file:
                if 'ESPERADOS=' in line:
                    result = line.split('ESPERADOS=')[1]
                    return result.strip()
    except FileNotFoundError:
        print("File 'pc.cfg' not found.")

# Função para remover ';' do texto
def remove_semicolon(text):
    return text.replace(';', '')

# Função para somar os números de uma string
def sum_digits(digit):
    return sum(int(x) for x in digit if x.isdigit())

# Fazendo o parse do arquivo
tree = ET.parse(read_leia(cfg))
root = tree.getroot()

# Abrindo o arquivo csv consultas
with open(read_consulta(cfg), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter = ';')
    
    # Escrevendo o nome das colunas
    writer.writerow(['QueryNumber', 'QueryText'])
    
    # Extraindo os elementos QueryNumber e QueryText
    for query in root.iter('QUERY'):
        query_number = int(query.find('QueryNumber').text)
        query_text = query.find('QueryText').text
        
        # Removendo os ';'
        query_text = remove_semicolon(query_text)

        # Escrevendo os elementos no arquivo
        writer.writerow([query_number, query_text])

# Abrindo o arquivo csv esperados
with open(read_esperado(cfg), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter = ';')
    
    # Escrevendo o nome das colunas
    writer.writerow(['QueryNumber', 'DocNumber','DocVotes'])
    
    # Extraindo os elementos QueryNumber e Item
    for query in root.iter('QUERY'):
        query_number = int(query.find('QueryNumber').text)

        for records in query.iter('Records'):
            for item in records.iter('Item'):
                item_value = item.text
                item_score = sum_digits(item.get('score')) #Faz a soma do número de votos
                

            # Escrevendo os elementos em um csv
                writer.writerow([query_number, item_value, item_score])

