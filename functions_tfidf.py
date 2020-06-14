from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf_model(train_docs):
    vectorizer = TfidfVectorizer()
    docs_tfidf = vectorizer.fit_transform(train_docs)
    return vectorizer, docs_tfidf


def get_most_similar_term(term_name, vectorizer, docs_tfidf, train_docs, train_set_term_mappings):
    query_tfidf = vectorizer.transform([term_name])
    cosine_similarities = list(cosine_similarity(query_tfidf, docs_tfidf).flatten())
    max_cosine_similarity_index = cosine_similarities.index(max(cosine_similarities))
    max_similar_train_term = train_docs[max_cosine_similarity_index]
    referent_hypothesis = train_set_term_mappings[max_similar_train_term]
    return referent_hypothesis, max(cosine_similarities)
