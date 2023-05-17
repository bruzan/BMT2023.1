# Importando as bibliotecas
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
import re
import csv

# Definindo o arquivo de configuração
cfg = 'config/gli.cfg'

# Função que retorna os caminhos dos arquivos da instrução LEIA
def read_leia(arquivo):
    config_data = []
    try:
        with open(arquivo, 'r') as file:
            for line in file:
                if 'LEIA=' in line:
                    config_data.append(line.split('LEIA=', 1)[1].strip())
        return config_data
    except FileNotFoundError:
        print("File 'gli.cfg' not found.")

# Função que retorna o caminho do arquivo da instrução ESCREVA
def read_escreva(arquivo):
    try:
        with open(arquivo, 'r') as file:
            for line in file:
                if 'ESCREVA=' in line:
                    result = line.split('ESCREVA=')[1]
                    return result.strip()
    except FileNotFoundError:
        print("File 'gli.cfg' not found.")

# Função para extrair o texto de um elemtno
def get_element_text(element):
    if element is not None:
        return element.text.strip()
    return ""

# Função que gera a lista invertida
def generate_inverted_list(xml_files, output_file):
    inverted_list = {}
    stop_words = set(stopwords.words('english'))
    
    # Iterando sobre os arquivos xml e fazendo o parse de cada um
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extraindo cada elemento RECORDNUM e ABSTRACT/EXTRACT
        for record in root.findall('.//RECORD'):
            record_num_elem = record.find('RECORDNUM')
            if record_num_elem is None:
                continue
            
            record_num = int(record_num_elem.text.strip())

            abstract_elem = record.find('ABSTRACT')
            extract_elem = record.find('EXTRACT')
            
            # Pegando o elemento EXTRACT caso não exista ABSTRACT
            if abstract_elem is not None:
                text = abstract_elem.text
            elif extract_elem is not None:
                text = extract_elem.text
            else:
                continue
            
            words = text.strip().split() # Separa os textos em listas de palavras

            
            # Remove os caracteres especiais
            for i in range(len(words)):
                #words[i] = re.sub('[!.()[]','',words[i])
                words[i] = re.sub('"(.,!?;:)[-]','',words[i])
            # Removendo as stopwords
            filtered_words = [word.upper() for word in words if word.lower() not in stop_words]
        
            # Construindo a lista invertida
            for word in filtered_words:
                word = word.upper().strip('".,!?;:()[]-')
                if word not in inverted_list:
                    inverted_list[word] = []
                inverted_list[word].append(record_num)
    
    # Escrevendo a lista em um arquivo csv
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for word, doc_ids in inverted_list.items():
            writer.writerow([word, doc_ids])


generate_inverted_list(read_leia(cfg), read_escreva(cfg))