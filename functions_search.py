from functions_text_processing import get_most_similar_term_jacard_ngrams, get_most_similar_term_jacard_tokens,\
    get_most_similar_term_jacard_average, get_most_similar_term_word_embedding_by_mean, get_most_similar_term_word_embedding_by_mean_threshold
from functions_huggingface import get_most_similar_term_biobert
from functions_word_embedding import get_bio_nlp_word_vec_model

dummy_hypothesis = 'OBT:002572'


def search_exact_match(train_set_term_mapping, term_name):
    if term_name in train_set_term_mapping:
        referent_hypothesis = train_set_term_mapping[term_name]
    else:
        referent_hypothesis = ''
    return referent_hypothesis


def search_exact_match_plus_onto_biotope(train_set_term_mapping, ontology_mapping, term_name):
    if term_name in train_set_term_mapping:
        referent_hypothesis = train_set_term_mapping[term_name]
    elif term_name in ontology_mapping:
        referent_hypothesis = ontology_mapping[term_name]
    else:
        referent_hypothesis = ''
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
        referent_hypothesis_train, max_similarity_train, most_similar_term_known= get_most_similar_term_jacard_average(term_name, train_set_term_mappings)
        referent_hypothesis_ontology, max_similarity_ontology, most_similar_term_known = get_most_similar_term_jacard_average(term_name, ontology_mappings)
        if max_similarity_train > max_similarity_ontology:
            referent_hypothesis = referent_hypothesis_train
        elif max_similarity_ontology > max_similarity_train:
            referent_hypothesis = referent_hypothesis_ontology
        else:
            referent_hypothesis = referent_hypothesis_ontology
    return referent_hypothesis


def search_exact_match_plus_onto_biotope_plus_biobert(train_set_term_mappings, term_name,
                                                          ontology_mappings,
                                                          train_set_term_name_embeddings,
                                                          ontology_term_name_embeddings,
                                                          bert_model, bert_tokenizer
                                                          ):

    if term_name in train_set_term_mappings:
        referent_hypothesis = train_set_term_mappings[term_name]
    elif term_name in ontology_mappings:
        referent_hypothesis = ontology_mappings[term_name]
    else:
        referent_hypothesis_train, max_similarity_train = get_most_similar_term_biobert(term_name,
                                                                                        train_set_term_mappings,
                                                                                        train_set_term_name_embeddings,
                                                                                        bert_model, bert_tokenizer)
        referent_hypothesis_ontology, max_similarity_ontology = get_most_similar_term_biobert(term_name,
                                                                                              ontology_mappings,
                                                                                              ontology_term_name_embeddings,
                                                                                              bert_model,
                                                                                              bert_tokenizer)
        if max_similarity_train > max_similarity_ontology:
            referent_hypothesis = referent_hypothesis_train
        elif max_similarity_ontology > max_similarity_train:
            referent_hypothesis = referent_hypothesis_ontology
        else:
            referent_hypothesis = referent_hypothesis_ontology

    return referent_hypothesis


def search_exact_match_plus_onto_biotope_plus_jacard_average_plus_rules(train_set_term_mappings, ontology_mappings, term_name):
    merged_term_name_mappings = {**train_set_term_mappings, **ontology_mappings}

    if term_name in merged_term_name_mappings:
        referent_hypothesis = merged_term_name_mappings[term_name]
    else:
        referent_hypothesis, max_similarity, most_similar_term_name = get_most_similar_term_jacard_average(term_name,
                                                                                                           merged_term_name_mappings)

        if 'rind' in term_name:
            referent_hypothesis = 'OBT:001481'  # cheese rind
        elif 'Polygenis' in term_name or 'Rhopalopsyllus' in term_name:
            referent_hypothesis = 'OBT:002034'  # flea
        elif 'Amblyomma' in term_name:
            referent_hypothesis = 'OBT:001821'  # tick
        elif 'sera' in term_name:
            referent_hypothesis = 'OBT:000524'  # blood serum
        elif 'PMN' in term_name:
            referent_hypothesis = 'OBT:001134'  # granulocyte
        elif 'patients with' in term_name:
            referent_hypothesis = 'OBT:003269'  # patient with infectious disease
        else:
            last_word = term_name.split()[-1]
            if 'units' in last_word:
                referent_hypothesis = 'OBT:000097'  # hospital environment
            elif 'surfaces' in last_word:
                referent_hypothesis = 'OBT:001341'  # surface of cheese

    return referent_hypothesis


def search_exact_match_plus_onto_biotope_plus_plus_rules_plus_word_embedding_by_mean(train_set_term_mappings, ontology_mappings, term_name, word_embedding_model):
    merged_term_name_mappings = {**train_set_term_mappings, **ontology_mappings}
    if term_name in train_set_term_mappings:
        referent_hypothesis = train_set_term_mappings[term_name]
    elif term_name in ontology_mappings:
        referent_hypothesis = ontology_mappings[term_name]
    else:
        term_words = term_name.split()
        last_word = term_words[-1]
        if 'units' in last_word:
            referent_hypothesis = 'OBT:000097'  # hospital environment
        elif 'group' in last_word:
            referent_hypothesis = 'OBT:003245'  # adult human
        elif 'surfaces' in last_word:
            referent_hypothesis = 'OBT:001341'  # surface of cheese
        elif 'cheese' in last_word and len(term_words) == 2:
            first_word = term_words[0]
            if first_word == 'Casera':
                referent_hypothesis = 'OBT:003483'  # valtellina casera
            else:
                try:
                    referent_hypothesis = ontology_mappings[first_word]  # type of cheese
                except:
                    referent_hypothesis = 'OBT:001480'  # cheese
        else:
            if 'rind' in term_name:
                referent_hypothesis = 'OBT:001481'  # cheese rind
            elif 'Polygenis' in term_name or 'Rhopalopsyllus' in term_name:
                referent_hypothesis = 'OBT:002034'  # flea
            elif 'sera' in term_name:
                referent_hypothesis = 'OBT:000524'  # blood serum
            elif 'Amblyomma' in term_name:
                referent_hypothesis = 'OBT:001821'  # tick
            elif 'PMN' in term_name:
                referent_hypothesis = 'OBT:001134'  # granulocyte
            elif 'patients with' in term_name:
                referent_hypothesis = 'OBT:003269'  # patient with infectious disease
            elif 'HMDM' in term_name or 'macrophage' in term_name:
                referent_hypothesis = 'OBT:002995'  # macrophage
            elif 'agar' in term_words or 'HCLA' in term_name or 'LPM' in term_name:
                referent_hypothesis = 'OBT:000031'  # agar
            else:
                referent_hypothesis, max_similarity, most_similar_term_name = get_most_similar_term_word_embedding_by_mean_threshold(
                    term_name, merged_term_name_mappings, word_embedding_model)

    return referent_hypothesis


