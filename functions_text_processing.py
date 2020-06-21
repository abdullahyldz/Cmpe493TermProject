import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import PorterStemmer
from functions_word_embedding import get_mean_word_embedding
from sklearn.metrics.pairwise import cosine_similarity
ps = PorterStemmer()


def get_most_similar_term_weighted_jacard_tokens(term_name, term_mappings, headwords):
    terms_known = list(term_mappings.keys())
    most_similar_term_known = terms_known[0]
    max_similarity = 0
    tokens_term_name = term_name.split()
    headword = headwords.split()
    for term_known in terms_known:
        tokens_term_known = term_known.split()
        new_similarity = weighted_jaccard_similarity(tokens_term_name, tokens_term_known, headword)
        if new_similarity > max_similarity:
            most_similar_term_known = term_known
            max_similarity = new_similarity

    most_similar_term_id = term_mappings[most_similar_term_known]
    return most_similar_term_id, max_similarity


def get_highest_jacard_average_score(term_name, term_mappings1, term_mappings2):
    id1, sim1 = get_most_similar_term_jacard_average(term_name, term_mappings1)
    id2, sim2 = get_most_similar_term_jacard_average(term_name, term_mappings2)
    if sim1 > sim2:
        return id1, sim1
    else:
        return id2, sim2


def weighted_jaccard_similarity(list1, list2, headword):  # works only for tokens
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        extra = len(set(list(set(list1).intersection(list2))).intersection(headword))
        if union == 0:
            return 0
        return float(intersection + extra) / union


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
    return most_similar_term_id, max_similarity, most_similar_term_known


def get_most_similar_term_word_embedding_by_mean(term_name, term_mappings, word_embedding_model):
    terms_known = list(term_mappings.keys())

    most_similar_term_known = terms_known[0]
    max_similarity = 0

    term_name_mean_word_embedding = get_mean_word_embedding(term_name, word_embedding_model)

    for term_known in terms_known:
        term_known_mean_word_embedding = get_mean_word_embedding(term_known, word_embedding_model)
        mean_word_embedding_similarity = cosine_similarity(term_name_mean_word_embedding.reshape(1, -1), term_known_mean_word_embedding.reshape(1,-1))[0][0]
        if mean_word_embedding_similarity > max_similarity:
            most_similar_term_known = term_known
            max_similarity = mean_word_embedding_similarity

    most_similar_term_id = term_mappings[most_similar_term_known]
    return most_similar_term_id, max_similarity, most_similar_term_known


def get_most_similar_term_word_embedding_by_mean_threshold(term_name, term_mappings, word_embedding_model):
    terms_known = list(term_mappings.keys())

    most_similar_term_known = terms_known[0]
    max_similarity = 0

    term_name_mean_word_embedding = get_mean_word_embedding(term_name, word_embedding_model)

    for term_known in terms_known:
        term_known_mean_word_embedding = get_mean_word_embedding(term_known, word_embedding_model)
        mean_word_embedding_similarity = cosine_similarity(term_name_mean_word_embedding.reshape(1, -1), term_known_mean_word_embedding.reshape(1,-1))[0][0]
        if mean_word_embedding_similarity > max_similarity:
            most_similar_term_known = term_known
            max_similarity = mean_word_embedding_similarity

    most_similar_term_id = term_mappings[most_similar_term_known]

    if max_similarity > 0.5:
        return most_similar_term_id, max_similarity, most_similar_term_known
    else:
        most_similar_term_id = ''
        max_similarity = 0
        most_similar_term_known = ''
        return most_similar_term_id, max_similarity, most_similar_term_known

