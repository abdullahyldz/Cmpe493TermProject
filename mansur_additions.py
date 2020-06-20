
# ADDED TO EXPERIMENTS
def experiment_exact_match_ontology_root_match_plus_jacard_average(train_set_term_mappings, dev_set_term_mappings, ontology_mappings):
    true_count = 0
    total_count = 0
    headword_match = 0
    headword_match_true = 0
    for term_name, referent_true in dev_set_term_mappings.items():
        headword = find_headword(term_name)
        total_count += 1
        head_match = False
        if term_name in ontology_mappings:
            referent_hypothesis = ontology_mappings[term_name]
        elif term_name in train_set_term_mappings:
            referent_hypothesis = train_set_term_mappings[term_name]

        else:
            referent_hypothesis_whole, max_similarity_whole = get_highest_jacard_average_score(term_name, train_set_term_mappings, ontology_mappings)
            referent_hypothesis_root, max_similarity_root = get_highest_jacard_average_score(headword,
                                                                                               train_set_term_mappings,
                                                                                               ontology_mappings)

            if max_similarity_whole > max_similarity_root:
                referent_hypothesis = referent_hypothesis_whole
            elif max_similarity_root > max_similarity_whole:
                referent_hypothesis = referent_hypothesis_root
            else:
                referent_hypothesis = referent_hypothesis_root

        if referent_hypothesis == referent_true:
            true_count += 1
            if head_match:
                headword_match_true += 1
        else:

            hypothesis_term_name = \
            [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_hypothesis][0]
            true_term_name = \
            [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_true][0]
            print('TermName: ' + term_name + '. Found: ' + hypothesis_term_name + '. Actual: ' + true_term_name)
    accuracy = true_count / total_count
    print(accuracy)


def experiment_exact_match_ontology_plus_weighted_jacard_average(train_set_term_mappings, dev_set_term_mappings, ontology_mappings):
    true_count = 0
    total_count = 0
    nlp = spacy.load("en_core_web_sm")
    for term_name, referent_true in dev_set_term_mappings.items():
        total_count += 1
        if term_name in ontology_mappings:
            referent_hypothesis = ontology_mappings[term_name]
        elif term_name in train_set_term_mappings:
            referent_hypothesis = train_set_term_mappings[term_name]
        else:
            headword = find_headword(term_name, nlp)
            referent_hypothesis_train, max_similarity_train = get_most_similar_term_weighted_jacard_tokens(term_name, train_set_term_mappings, headword)
            referent_hypothesis_onto, max_similarity_onto = get_most_similar_term_weighted_jacard_tokens(term_name, ontology_mappings, headword)

            if max_similarity_train > max_similarity_onto:
                referent_hypothesis = referent_hypothesis_train
            elif max_similarity_onto > max_similarity_train:
                referent_hypothesis = referent_hypothesis_onto
            else:
                referent_hypothesis = referent_hypothesis_onto

        if referent_hypothesis == referent_true:
            true_count += 1

        else:

            hypothesis_term_name = \
            [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_hypothesis][0]
            true_term_name = \
            [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_true][0]
            print('TermName: ' + term_name + '. Found: ' + hypothesis_term_name + '. Actual: ' + true_term_name)
    accuracy = true_count / total_count
    print(accuracy)

# ADDED TO TEXT PROCESSING

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