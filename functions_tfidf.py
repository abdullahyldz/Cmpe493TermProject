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


def experiment_exact_match_plus_onto_biotope_plus_tfidf(train_set_term_mappings, dev_set_term_mappings, ontology_mapping):
    true_count = 0
    total_count = 0

    train_docs = list(train_set_term_mappings.keys())
    vectorizer, docs_tfidf = fit_tfidf_model(train_docs)

    for term_name, referent_true in dev_set_term_mappings.items():
        total_count += 1
        if term_name in train_set_term_mappings:
            referent_hypothesis = train_set_term_mappings[term_name]
        elif term_name in ontology_mapping:
            referent_hypothesis = ontology_mapping[term_name]
        else:
            referent_hypothesis, max_value = get_most_similar_term(term_name, vectorizer, docs_tfidf, train_docs, train_set_term_mappings)

        if referent_hypothesis == referent_true:
                true_count += 1
        else:
            print(term_name)
    accuracy = true_count/total_count
    print(accuracy)


def experiment_exact_match_plus_onto_biotope_plus_tfidf_plus_ontology_tfidf(train_set_term_mappings, dev_set_term_mappings, ontology_mappings):
    true_count = 0
    total_count = 0

    train_docs = list(train_set_term_mappings.keys())
    vectorizer_train, docs_tfidf_train = fit_tfidf_model(train_docs)

    ontology_docs = list(ontology_mappings.keys())
    vectorizer_ontology, docs_tfidf_ontology = fit_tfidf_model(ontology_docs)

    for term_name, referent_true in dev_set_term_mappings.items():
        total_count += 1
        if term_name in train_set_term_mappings:
            referent_hypothesis = train_set_term_mappings[term_name]
        elif term_name in ontology_mappings:
            referent_hypothesis = ontology_mappings[term_name]
        else:
            referent_hypothesis_train, max_value_train = get_most_similar_term(term_name, vectorizer_train, docs_tfidf_train, train_docs, train_set_term_mappings)
            referent_hypothesis_ontology, max_value_ontology = get_most_similar_term(term_name, vectorizer_ontology, docs_tfidf_ontology, ontology_docs, ontology_mappings)
            if max_value_train > max_value_ontology:
                referent_hypothesis = referent_hypothesis_train
            elif max_value_ontology > max_value_train:
                referent_hypothesis = referent_hypothesis_ontology
            else:
                referent_hypothesis = referent_hypothesis_ontology

        if referent_hypothesis == referent_true:
                true_count += 1
        else:
            hypothesis_term_name = list(ontology_mappings.keys())[list(ontology_mappings.values()).index(referent_hypothesis)]
            true_term_name = list(ontology_mappings.keys())[list(ontology_mappings.values()).index(referent_true)]
            print('TermName: ' + term_name + '. Found: ' + hypothesis_term_name + '. Actual: ' + true_term_name)
    accuracy = true_count/total_count
    print(accuracy)
