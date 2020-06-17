from gensim.test.utils import datapath
from gensim.models import KeyedVectors


def get_bio_word_vec():
    path = r"C:\Users\Can Deveci\Desktop\Biomedical Word Embeddings\BioWordVec_PubMed_MIMICIII_d200.vec.bin"
    model = KeyedVectors.load_word2vec_format(datapath(path), binary=True)
    word_vector = model.get_word_vector('bacteria')
    print(word_vector)


def bio_nlp_word_vec():
    path = r"C:\Users\Can Deveci\Desktop\Biomedical Word Embeddings\PubMed-shuffle-win-2.bin"
    wv_from_bin = KeyedVectors.load_word2vec_format(datapath(path), binary=True)  # C bin format
    word_embedding = wv_from_bin.wv['bacteria']
    return word_embedding


# word_embedding = bio_nlp_word_vec()  # works
get_bio_word_vec()




