# Carregando bibliotecas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import numpy as np
import operator

# Carregando todos os resultados
f_resultados = './Trabalho 2/Results/resultados.csv'
f_resultados_ps = './Trabalho 2/Results/resultados_stemmer.csv'
f_esperados = './Trabalho 2/Results/esperados.csv'

queries = {}  # Sem stemmer
for line in open(f_resultados, 'r'):
    line = line.rstrip('\n')
    query = int(line.split(';')[0])
    result = line.split(';')[1].lstrip('[').rstrip(']')\
                .replace(' ', '').split(',')
    if query not in queries:
        queries[query] = {}
    if int(result[1]) not in queries[query]:
            queries[query][int(result[1])] = float(result[2])

queries_ps = {}  # Com stemmer
for line in open(f_resultados_ps, 'r'):
    line = line.rstrip('\n')
    query = int(line.split(';')[0])
    result = line.split(';')[1].lstrip('[').rstrip(']')\
                .replace(' ', '').split(',')
    if query not in queries_ps:
        queries_ps[query] = {}
    if int(result[1]) not in queries_ps[query]:
        queries_ps[query][int(result[1])] = float(result[2])

esperados = {}  # Esperados
for line in open(f_esperados, 'r'):
    if line.lower() != 'querynumber;docnumber;docvotes\n':
        line = line.rstrip('\n')
        if int(line.split(';')[0]) not in esperados:
            esperados[int(line.split(';')[0])] = {}
        if int(line.split(';')[1]) not in esperados[int(line.split(';')[0])]:
            esperados[int(line.split(';')[0])][int(line.split(';')[1])] = \
                int(line.split(';')[2])

def relevant(doc, esperados):
    if doc in esperados:
        return esperados[doc]
    else:
        return 0

# Calcula quantidade de verdadeiros positivos
def true_positive(a, b):
    tp = 0
    for item in a:
        if item in b:
            tp += 1
    return tp

# Calcula quantidade de falsos positivos
def false_positive(a, b):
    fp = 0
    for item in a:
        if item not in b:
            fp += 1
    return fp

# Retorna os documentos retornados em todas as consultas
def num_docs(a):
    full_prediction_set = {}
    for item in a:
        for doc in a[item]:
            if doc not in full_prediction_set:
                full_prediction_set[doc] = 0
            full_prediction_set[doc] += 1
    return full_prediction_set

# Calcula a quantidade de verdadeiros negativos
def true_negative(a, b, full_set):
    tn = 0
    neg = {}
    full_prediction_set = num_docs(full_set)
    for item in full_prediction_set:
        if item not in b:
            neg[item] = 1

    for item in neg:
        if item not in a:
            tn += 1
    return tn

# Calcula a quantidade de falsos negativos
def false_negative(a, b):
    fn = 0
    for item in b:
        if item not in a:
            fn += 1
    return fn

# Calcula recall
def get_recall(results, relevants, max_rank):
    count = 0
    considered_results = []
    for doc in results:
        count += 1
        if count <= max_rank:
            considered_results.append(doc)
        else:
            break

    TP = true_positive(considered_results, relevants)
    recall = TP / len(relevants)

    return recall

# Calcula precisão
def get_precision(results, relevants, max_rank):
    count = 0
    considered_results = []
    for doc in results:
        count += 1
        if count <= max_rank:
            considered_results.append(doc)
        else:
            break

    TP = true_positive(considered_results, relevants)
    if len(considered_results) > 0:
        precision = TP / len(considered_results)
    else:
        precision = 0.0

    return precision

def interpolation(queries, recall_levels, esperados):
    precision = {}
    for level in range(0, len(recall_levels)):
        recall_level = recall_levels[level]
        precision[recall_level] = {}
        for query in queries:
            max_results = int(len(esperados[query]) * recall_level)
            precision[recall_level][query] = get_precision(queries[query],
                                                           esperados[query],
                                                           max_results)
    for query in queries:
        for level in range(0, len(recall_levels)):
            if precision[recall_levels[level]][query] == 0:
                for r in range(level, len(recall_levels)):
                    if precision[recall_levels[r]][query] > \
                       precision[recall_levels[level]][query]:
                        precision[recall_levels[level]][
                            query] = precision[recall_levels[r]][query]
    avg_precision = {}
    for level in recall_levels:
        avg_precision[level] = 0
        for query in queries:
            avg_precision[level] += precision[level][query]
        avg_precision[level] /= len(queries)
    return avg_precision

# Precisão@
def precision_at(queries, esperados, rank):
    precision = {}
    for query in queries:
        precision[query] = get_precision(queries[query],
                                         esperados[query],
                                         rank)
    avg_precision = 0
    for query in queries:
        avg_precision += precision[query]
    avg_precision /= len(queries)
    return avg_precision

# F1
def f1score_at(queries, esperados, recall_level):
    f1 = {}
    for query in queries:
        max_rank = int(len(esperados[query]) * recall_level)
        precision = get_precision(queries[query], esperados[query], max_rank)
        recall = get_recall(queries[query], esperados[query], max_rank)
        if (precision + recall) > 0:
            f1[query] = 2 * (precision * recall) / (precision + recall)
        else:
            f1[query] = 0
    return f1

# Média
def avg(list):
    mean, count = 0, 0
    for i in list:
        count += 1
        mean += list[i]
    mean /= count
    return mean

# MRR
def mrr(queries, esperados):
    ReciprocalRank = {}
    for query in queries:
        rank = 0
        for doc in queries[query]:
            rank += 1
            if doc in esperados[query]:
                break
        ReciprocalRank[query] = 1 / rank
    mean_ReciprocalRank = 0
    for query in queries:
        mean_ReciprocalRank += ReciprocalRank[query]
    mean_ReciprocalRank /= len(queries)
    return mean_ReciprocalRank

# Discounted Cumulative Gain
def dcg(queries, esperados):
    DiscountedCumulativeGain = {}
    for query in queries:
        dcg = 0
        rank = 0
        for doc in queries[query]:
            rank += 1
            if doc in esperados[query]:
                dcg += esperados[query][doc] / (math.log2(rank + 1))
        DiscountedCumulativeGain[query] = dcg
    mean_DiscountedCumulativeGain = 0
    for query in queries:
        mean_DiscountedCumulativeGain += DiscountedCumulativeGain[query]
    mean_DiscountedCumulativeGain /= len(queries)
    return (mean_DiscountedCumulativeGain, DiscountedCumulativeGain)

def ndcg(queries, esperados):
    nDiscountedCumulativeGain = {}
    for query in queries:
        idcg = 0
        esperados_sorted = sorted(esperados[query].items(),
                                  key = operator.itemgetter(1),
                                  reverse = True)
        for i in range(0, len(esperados_sorted)):
            rank = i + 1
            idcg += ((2 ^ esperados_sorted[i][1])-1) / (math.log2(rank + 1))
        dcg = 0
        rank = 0
        for doc in queries[query]:
            rank += 1
            if doc in esperados[query]:
                dcg += ((2 ^ esperados[query][doc])-1) / (math.log2(rank + 1))
        nDiscountedCumulativeGain[query] = dcg / idcg
    mean_nDiscountedCumulativeGain = 0
    for query in queries:
        mean_nDiscountedCumulativeGain += nDiscountedCumulativeGain[query]
    mean_nDiscountedCumulativeGain /= len(queries)
    return (mean_nDiscountedCumulativeGain, nDiscountedCumulativeGain)



# Gráfico de 11 pontos de precisão e recall

recall_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
noStemmerRP = interpolation(queries, recall_levels, esperados)
stemmerRP = interpolation(queries_ps, recall_levels, esperados)
plt.figure(1)
plt.plot(*zip(*sorted(noStemmerRP.items())), label = 'Sem stemmer')
plt.plot(*zip(*sorted(stemmerRP.items())), label = 'Com stemmer')
plt.xlabel('Recall')
plt.ylabel('Precisão')
plt.title('Recall-Precisão')
plt.legend()
plt.savefig('F:/BMT/Trabalho 3 - Avaliação/Plots/11pontos-stemmerANDnostemmer.png')

# F1-Score
F1_nostem = f1score_at(queries, esperados, 0.7)
F1_stem = f1score_at(queries_ps, esperados, 0.7)
print('F1-score @ 70% recall without PorterStemer: ' +
      str(avg(F1_nostem)))
print('F1-score @ 70% recall with PorterStemer: ' +
      str(avg(F1_stem)))


# Precisão@5 e @10

P_5_nostem = precision_at(queries, esperados, 5)
print('Precision @ 5 sem stemmer: ' + str(P_5_nostem))
P_10_nostem = precision_at(queries, esperados, 10)
print('Precision @ 10 sem stemmer: ' + str(P_10_nostem))
P_5_stem = precision_at(queries_ps, esperados, 5)
print('Precision @ 5 com stemmer: ' + str(P_5_stem))
P_10_stem = precision_at(queries_ps, esperados, 10)
print('Precision @ 10 com stemmer: ' + str(P_10_stem))


# R-Precision

RP = {}
for query in queries:
    max_rank = len(esperados[query])
    noStemmerP = get_precision(queries[query], esperados[query], max_rank)
    PorterStemmerP = get_precision(queries_ps[query], esperados[query],
                                   max_rank)
    RP[query] = PorterStemmerP - noStemmerP
plt.figure(2)
plt.gca().yaxis.grid()
plt.gca().xaxis.grid()
rect = plt.bar(*zip(*sorted(RP.items())), align='center')
plt.yticks(np.arange(-0.4, 0.5, 0.1))
plt.xlabel('Consulta')
plt.ylabel('Recall-Precision')
plt.title('Recall-Precision por consulta')
a_patch = mpatches.Patch(label='Recall @ 100%')
b_patch = mpatches.Patch(label='Stemmer - noStemmer')
plt.legend(handles=[a_patch, b_patch])
plt.savefig('F:/BMT/Trabalho 3 - Avaliação/Plots/R-Precision-StemmerANDnostemmer.png')

# MAP

MAP_nostem = avg(noStemmerRP)
MAP_stem = avg(stemmerRP)
print('MAP sem stemmer: ' + str(MAP_nostem))
print('MAP com stemmer: ' + str(MAP_stem))

# MRR

MRR_nostem = mrr(queries, esperados)
print('MRR sem stemmer: ' + str(MRR_nostem))
MRR_stem = mrr(queries_ps, esperados)
print('MRR com stemmer: ' + str(MRR_stem))


# Discounted Cumulative Gain (Médio)

dcg_nostem = dcg(queries, esperados)
print('Discounted Cumulative Gain sem stemmer = ' +
      str(dcg_nostem[0]))

dcg_stem = dcg(queries_ps, esperados)
print('Discounted Cumulative Gain com stemmer = ' +
      str(dcg_stem[0]))

plt.figure(3)
plt.bar(*zip(*sorted(dcg_nostem[1].items())),
        label = 'Sem stemmer')
plt.bar(*zip(*sorted(dcg_stem[1].items())),
        label = 'Com stemmer')
plt.xlabel('Queries')
plt.ylabel('Discounted Cumulative Gain')
plt.title('Discounted Cumulative Gain por consulta')
plt.legend()
plt.savefig('F:/BMT/Trabalho 3 - Avaliação/Plots/DCG-stemmerANDnostemmer.png')

# Normalized DCG - Discounted Cumulative Gain
ndcg_nostem = ndcg(queries, esperados)
print('Normalized discounted cumulative gain sem stemmer = ' +
      str(ndcg_nostem[0]))

ndcg_stem = ndcg(queries_ps, esperados)
print('Normalized discounted cumulative gain com stemmer = ' +
      str(ndcg_stem[0]))


plt.figure(4)
plt.bar(*zip(*sorted(ndcg_nostem[1].items())),
        label = 'Sem stemmer')
plt.bar(*zip(*sorted(ndcg_stem[1].items())),
        label = 'Com stemmer')
plt.xlabel('Queries')
plt.ylabel('Normalized Discounted Cumulative Gain')
plt.title('Normalized Discounted Cumulative Gain por consulta')
plt.legend()
plt.savefig('F:/BMT/Trabalho 3 - Avaliação/Plots/normalizedDCG-stemmerANDnostemmer.png')


