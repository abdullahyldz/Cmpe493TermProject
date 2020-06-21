from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import numpy as np


def get_bio_word_vec():
    path = r"C:\Users\Can Deveci\Desktop\Biomedical Word Embeddings\BioWordVec_PubMed_MIMICIII_d200.vec.bin"
    model = KeyedVectors.load_word2vec_format(datapath(path), binary=True)
    word_vector = model.get_word_vector('bacteria')
    print(word_vector)


def get_bio_nlp_word_vec_model():
    path = r"C:\Users\Can Deveci\Desktop\Biomedical Word Embeddings\PubMed-shuffle-win-2.bin"
    wv_from_bin = KeyedVectors.load_word2vec_format(datapath(path), binary=True)  # C bin format
    return wv_from_bin


def get_bio_nlp_word_vec(word, word_embedding_model):
    word_embedding = word_embedding_model.wv[word]
    return word_embedding


def get_mean_word_embedding(term_name, word_embedding_model):
    term_words = term_name.split()

    mean_word_embedding = np.zeros(200)

    for term_word in term_words:
        try:
            term_word_embedding = get_bio_nlp_word_vec(term_word, word_embedding_model)
        except:
            continue
        mean_word_embedding += term_word_embedding
    mean_word_embedding = mean_word_embedding/len(term_words)
    return mean_word_embedding


