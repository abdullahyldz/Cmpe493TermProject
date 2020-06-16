from functions_text_processing import get_most_similar_term_jacard_ngrams, get_most_similar_term_jacard_tokens,\
    get_most_similar_term_jacard_average


dummy_hypothesis = 'OBT:002572'


def search_exact_match(train_set_term_mapping, term_name):
    if term_name in train_set_term_mapping:
        referent_hypothesis = train_set_term_mapping[term_name]
        return referent_hypothesis
    else:
        return dummy_hypothesis


def search_exact_match_plus_onto_biotope(train_set_term_mapping, ontology_mapping, term_name):
    if term_name in train_set_term_mapping:
        referent_hypothesis = train_set_term_mapping[term_name]
    elif term_name in ontology_mapping:
        referent_hypothesis = ontology_mapping[term_name]
    else:
        referent_hypothesis = dummy_hypothesis
    return referent_hypothesis


def search_exact_match_plus_onto_biotope_plus_jacard_ngrams(train_set_term_mappings, ontology_mappings, term_name):
    if term_name in train_set_term_mappings:
        referent_hypothesis = train_set_term_mappings[term_name]
    elif term_name in ontology_mappings:
        referent_hypothesis = ontology_mappings[term_name]
    else:
        referent_hypothesis_train, max_similarity_train = get_most_similar_term_jacard_ngrams(term_name,
                                                                                              train_set_term_mappings)
        referent_hypothesis_ontology, max_similarity_ontology = get_most_similar_term_jacard_ngrams(term_name,
                                                                                                    ontology_mappings)
        if max_similarity_train > max_similarity_ontology:
            referent_hypothesis = referent_hypothesis_train
        elif max_similarity_ontology > max_similarity_train:
            referent_hypothesis = referent_hypothesis_ontology
        else:
            referent_hypothesis = referent_hypothesis_ontology
    return referent_hypothesis


def search_exact_match_plus_onto_biotope_plus_jacard_tokens(train_set_term_mappings, ontology_mappings, term_name):
    if term_name in train_set_term_mappings:
        referent_hypothesis = train_set_term_mappings[term_name]
    elif term_name in ontology_mappings:
        referent_hypothesis = ontology_mappings[term_name]
    else:
        referent_hypothesis_train, max_similarity_train = get_most_similar_term_jacard_tokens(term_name,
                                                                                              train_set_term_mappings)
        referent_hypothesis_ontology, max_similarity_ontology = get_most_similar_term_jacard_tokens(term_name,
                                                                                                    ontology_mappings)
        if max_similarity_train > max_similarity_ontology:
            referent_hypothesis = referent_hypothesis_train
        elif max_similarity_ontology > max_similarity_train:
            referent_hypothesis = referent_hypothesis_ontology
        else:
            referent_hypothesis = referent_hypothesis_ontology
    return referent_hypothesis


def search_exact_match_plus_onto_biotope_plus_jacard_average(train_set_term_mappings, ontology_mappings, term_name):
    if term_name in train_set_term_mappings:
        referent_hypothesis = train_set_term_mappings[term_name]
    elif term_name in ontology_mappings:
        referent_hypothesis = ontology_mappings[term_name]
    else:
        referent_hypothesis_train, max_similarity_train = get_most_similar_term_jacard_average(term_name,
                                                                                               train_set_term_mappings)
        referent_hypothesis_ontology, max_similarity_ontology = get_most_similar_term_jacard_average(term_name,
                                                                                                     ontology_mappings)
        if max_similarity_train > max_similarity_ontology:
            referent_hypothesis = referent_hypothesis_train
        elif max_similarity_ontology > max_similarity_train:
            referent_hypothesis = referent_hypothesis_ontology
        else:
            referent_hypothesis = referent_hypothesis_ontology
    return referent_hypothesis

