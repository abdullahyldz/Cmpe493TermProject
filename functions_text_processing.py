import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import PorterStemmer
ps = PorterStemmer()

def lemmatize_term(term):
    words = term.split()
    lemma = nltk.wordnet.WordNetLemmatizer()
    new_words = []
    for i in words:
        new_words.append(lemma.lemmatize(i))
    new_term = " ".join(new_words)
    return new_term


def remove_stop_words(term):
    stop = stopwords.words('english')
    words = term.split()
    k = 0
    while k < len(words):
        if words[k] in stop:
            words.remove(words[k])
            k = k - 1
        k = k + 1
    return words


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection

    if union == 0:
        return 0
    return float(intersection) / union

def jaccard_similarity_stemming(list1, list2):
    #TODO stemming before comparison
    list1 = [ps.stem(elem) for elem in list1]
    list2 = [ps.stem(elem) for elem in list2]

    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection

    if union == 0:
        return 0
    return float(intersection) / union

def get_most_similar_term_jacard_ngrams(term_name, term_mappings):
    ngram_count = 2
    terms_known = list(term_mappings.keys())

    most_similar_term = terms_known[0]
    max_similarity = 0
    ngrams_term_name = list(ngrams(term_name, ngram_count))

    for term_known in terms_known:
        ngrams_term_known = list(ngrams(term_known, ngram_count))
        new_similarity = jaccard_similarity(ngrams_term_name, ngrams_term_known)
        if new_similarity > max_similarity:
            most_similar_term = term_known
            max_similarity = new_similarity

    most_similar_term_id = term_mappings[most_similar_term]
    return most_similar_term_id, max_similarity


def get_most_similar_term_jacard_tokens(term_name, term_mappings):
    terms_known = list(term_mappings.keys())
    most_similar_term_known = terms_known[0]
    max_similarity = 0
    tokens_term_name = term_name.split()

    for term_known in terms_known:
        tokens_term_known = term_known.split()
        new_similarity = jaccard_similarity(tokens_term_name, tokens_term_known)
        if new_similarity > max_similarity:
            most_similar_term_known = term_known
            max_similarity = new_similarity

    most_similar_term_id = term_mappings[most_similar_term_known]
    return most_similar_term_id, max_similarity


def get_most_similar_term_jacard_average(term_name, term_mappings):
    ngram_count = 2
    terms_known = list(term_mappings.keys())
    most_similar_term_known = terms_known[0]
    max_similarity = 0
    tokens_term_name = term_name.split()
    ngrams_term_name = list(ngrams(term_name, ngram_count))

    for term_known in terms_known:
        tokens_term_known = term_known.split()
        ngrams_term_known = list(ngrams(term_known, ngram_count))
        jacard_similarity_term = jaccard_similarity(tokens_term_name, tokens_term_known)
        jacard_similarity_ngram = jaccard_similarity(ngrams_term_name, ngrams_term_known)
        jacard_similarity_average = (jacard_similarity_term + jacard_similarity_ngram)/2
        if jacard_similarity_average > max_similarity:
            most_similar_term_known = term_known
            max_similarity = jacard_similarity_average

    most_similar_term_id = term_mappings[most_similar_term_known]
    return most_similar_term_id, max_similarity


