#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.fasttext import FastText
from rank_bm25 import BM25Okapi


# FUNCIÓN PARA GENERAR MODELO
def train_model(data, size):
    ft_model = FastText(
        sg=1,  # use skip-gram: usually gives better results
        size=size,  # embedding dimension
        window=5,  # window size: 5 tokens before and 5 tokens after to get wider context
        min_count=1,  # only consider tokens with at least n occurrences in the corpus
        negative=15,  # negative subsampling: bigger than default to sample negative examples more
        min_n=2,  # min character n-gram
        max_n=5,  # max character n-gram
        workers=2
    )
    ft_model.build_vocab(data.tokens.tolist())
    ft_model.train(
        data.tokens.tolist(),
        epochs=6,
        total_examples=ft_model.corpus_count,
        total_words=ft_model.corpus_total_words,
        queue_factor=1
    )
    return ft_model


# Calculamos los pesos para la indexación
def calculate_weights(data, model):
    weighted_doc_vects = []
    bm25 = BM25Okapi(data.tokens.tolist())
    for i, doc in enumerate(data.tokens.tolist()):
        doc_vector = []
        for word in doc:
            vector = model.wv[word]
            weight = (bm25.idf[word] * ((bm25.k1 + 1.0)*bm25.doc_freqs[i][word])) / (bm25.k1 * (
                1.0 - bm25.b + bm25.b * (bm25.doc_len[i]/bm25.avgdl))+bm25.doc_freqs[i][word])
            weighted_vector = vector * weight
            doc_vector.append(weighted_vector)

        weighted_doc_vects.append(np.mean(doc_vector, axis=0))
    return weighted_doc_vects


# train_df = pd.read_csv("train_fasttext.csv")  # .drop('Unnamed: 0', axis=1)
train_df = pd.read_csv("clean_vacancies_20210712.csv").drop(
    'Unnamed: 0', axis=1)
train_df["tokens"] = train_df.data.apply(lambda x: x.split(' '))

model = train_model(train_df, 100)
model.save('model/fasttext_model_100.bin')
weights = calculate_weights(train_df, model)
data_df = pd.DataFrame(np.concatenate([[array] for array in weights]))
data_df.to_csv("model/fasttext_weights_100.csv")

# model_300 = train_model(train_df, 300)
# model_300.save('model/fasttext_model_300.bin')
# weights_300 = calculate_weights(train_df, model_300)
# data_df_300 = pd.DataFrame(np.concatenate([[array] for array in weights_300]))
# data_df_300.to_csv("model/fasttext_weights_300.csv")
