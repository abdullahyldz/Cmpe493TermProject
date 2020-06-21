from functions_tfidf import fit_tfidf_model, get_most_similar_term
from functions_text_processing import get_most_similar_term_jacard_ngrams, get_most_similar_term_jacard_tokens,\
    get_most_similar_term_jacard_average, get_highest_jacard_average_score, get_most_similar_term_weighted_jacard_tokens,\
    get_most_similar_term_word_embedding_by_mean
from functions_huggingface import get_most_similar_term_biobert
from functions_pos_tagging import compareTags, createTags,find_headword
from functions_word_embedding import get_bio_nlp_word_vec_model
import spacy


def experiment_exact_match(train_set_term_mapping, dev_set_term_mapping):
    true_count = 0
    total_count = 0
    for term_name, referent_true in dev_set_term_mapping.items():
        total_count += 1
        if term_name in train_set_term_mapping:
            referent_hypothesis = train_set_term_mapping[term_name]
            if referent_hypothesis == referent_true:
                true_count += 1
        else:
            print(term_name)
    accuracy = true_count/total_count
    print(accuracy)


def experiment_exact_match_plus_onto_biotope(train_set_term_mappings, dev_set_term_mappings, ontology_mapping):
    true_count = 0
    total_count = 0
    for term_name, referent_true in dev_set_term_mappings.items():
        total_count += 1
        if term_name in train_set_term_mappings:
            referent_hypothesis = train_set_term_mappings[term_name]
        elif term_name in ontology_mapping:
            referent_hypothesis = ontology_mapping[term_name]

        if referent_hypothesis == referent_true:
                true_count += 1
        else:
            print(term_name)
    accuracy = true_count/total_count
    print(accuracy)


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


def experiment_exact_match_plus_onto_biotope_plus_tfidf_plus_onto_tfidf(train_set_term_mappings, dev_set_term_mappings, ontology_mappings):
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


def experiment_exact_match_plus_onto_biotope_plus_jacard_ngrams(train_set_term_mappings, dev_set_term_mappings, ontology_mappings):
    true_count = 0
    total_count = 0

    for term_name, referent_true in dev_set_term_mappings.items():
        total_count += 1
        if term_name == 'mature cheeses':
            print('')
        if term_name in train_set_term_mappings:
            referent_hypothesis = train_set_term_mappings[term_name]
        elif term_name in ontology_mappings:
            referent_hypothesis = ontology_mappings[term_name]
        else:
            referent_hypothesis_train, max_similarity_train = get_most_similar_term_jacard_ngrams(term_name, train_set_term_mappings)
            referent_hypothesis_ontology, max_similarity_ontology = get_most_similar_term_jacard_ngrams(term_name, ontology_mappings)
            if max_similarity_train > max_similarity_ontology:
                referent_hypothesis = referent_hypothesis_train
            elif max_similarity_ontology > max_similarity_train:
                referent_hypothesis = referent_hypothesis_ontology
            else:
                referent_hypothesis = referent_hypothesis_ontology

        if referent_hypothesis == referent_true:
                true_count += 1
        else:

            hypothesis_term_name = [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_hypothesis][0]
            true_term_name = [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_true][0]
            print('TermName: ' + term_name + '. Found: ' + hypothesis_term_name + '. Actual: ' + true_term_name)
    accuracy = true_count/total_count
    print(accuracy)


def experiment_exact_match_plus_onto_biotope_plus_jacard_tokens(train_set_term_mappings, dev_set_term_mappings, ontology_mappings):
    true_count = 0
    total_count = 0

    for term_name, referent_true in dev_set_term_mappings.items():
        total_count += 1
        if term_name in train_set_term_mappings:
            referent_hypothesis = train_set_term_mappings[term_name]
        elif term_name in ontology_mappings:
            referent_hypothesis = ontology_mappings[term_name]
        else:
            referent_hypothesis_train, max_similarity_train = get_most_similar_term_jacard_tokens(term_name, train_set_term_mappings)
            referent_hypothesis_ontology, max_similarity_ontology = get_most_similar_term_jacard_tokens(term_name, ontology_mappings)
            if max_similarity_train > max_similarity_ontology:
                referent_hypothesis = referent_hypothesis_train
            elif max_similarity_ontology > max_similarity_train:
                referent_hypothesis = referent_hypothesis_ontology
            else:
                referent_hypothesis = referent_hypothesis_ontology

        if referent_hypothesis == referent_true:
                true_count += 1
        else:

            hypothesis_term_name = [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_hypothesis][0]
            true_term_name = [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_true][0]
            print('TermName: ' + term_name + '. Found: ' + hypothesis_term_name + '. Actual: ' + true_term_name)
    accuracy = true_count/total_count
    print(accuracy)


def experiment_exact_match_plus_onto_biotope_plus_jacard_average(train_set_term_mappings, dev_set_term_mappings, ontology_mappings):
    true_count = 0
    total_count = 0

    for term_name, referent_true in dev_set_term_mappings.items():
        total_count += 1
        if term_name == 'mature cheeses':
            print('')
        if term_name in train_set_term_mappings:
            referent_hypothesis = train_set_term_mappings[term_name]
        elif term_name in ontology_mappings:
            referent_hypothesis = ontology_mappings[term_name]
        else:
            referent_hypothesis_train, max_similarity_train = get_most_similar_term_jacard_average(term_name, train_set_term_mappings)
            referent_hypothesis_ontology, max_similarity_ontology = get_most_similar_term_jacard_average(term_name, ontology_mappings)
            if max_similarity_train > max_similarity_ontology:
                referent_hypothesis = referent_hypothesis_train
            elif max_similarity_ontology > max_similarity_train:
                referent_hypothesis = referent_hypothesis_ontology
            else:
                referent_hypothesis = referent_hypothesis_ontology

        if referent_hypothesis == referent_true:
                true_count += 1
        else:

            hypothesis_term_name = [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_hypothesis][0]
            true_term_name = [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_true][0]
            print('TermName: ' + term_name + '. Found: ' + hypothesis_term_name + '. Actual: ' + true_term_name)
    accuracy = true_count/total_count
    print(accuracy)


def experiment_exact_match_plus_onto_biotope_plus_biobert(train_set_term_mappings, dev_set_term_mappings,
                                                          ontology_mappings,
                                                          train_set_term_name_embeddings,
                                                          ontology_term_name_embeddings,
                                                          bert_model, bert_tokenizer
                                                          ):
    true_count = 0
    total_count = 0

    for term_name, referent_true in dev_set_term_mappings.items():
        total_count += 1
        if term_name in train_set_term_mappings:
            referent_hypothesis = train_set_term_mappings[term_name]
        elif term_name in ontology_mappings:
            referent_hypothesis = ontology_mappings[term_name]
        else:
            referent_hypothesis_train, max_similarity_train = get_most_similar_term_biobert(term_name,
                                                                                            train_set_term_mappings,
                                                                                            train_set_term_name_embeddings,
                                                                                            bert_model, bert_tokenizer)
            referent_hypothesis_ontology, max_similarity_ontology = get_most_similar_term_biobert(term_name, ontology_mappings,
                                                                                                  ontology_term_name_embeddings,
                                                                                                  bert_model, bert_tokenizer)
            if max_similarity_train > max_similarity_ontology:
                referent_hypothesis = referent_hypothesis_train
            elif max_similarity_ontology > max_similarity_train:
                referent_hypothesis = referent_hypothesis_ontology
            else:
                referent_hypothesis = referent_hypothesis_ontology

        if referent_hypothesis == referent_true:
                true_count += 1
        else:

            hypothesis_term_name = [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_hypothesis][0]
            true_term_name = [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_true][0]
            print('TermName: ' + term_name + '. Found: ' + hypothesis_term_name + '. Actual: ' + true_term_name)
    accuracy = true_count/total_count
    print(accuracy)


def experiment_exact_match_plus_onto_biotope_plus_pos_tagger(train_set_term_mappings, dev_set_term_mappings, ontology_mappings):
    true_count = 0
    total_count = 0
    ################################################################################
    train_postags, ontology_postags = createTags(train_set_term_mappings, ontology_mappings)
    for term_name, referent_true in dev_set_term_mappings.items():
        total_count += 1
        if term_name in train_set_term_mappings:
            referent_hypothesis = train_set_term_mappings[term_name]
        elif term_name in ontology_mappings:
            referent_hypothesis = ontology_mappings[term_name]
        else:
            referent_hypothesis_train, max_similarity_train = get_most_similar_term_jacard_ngrams(term_name,train_set_term_mappings)
            referent_hypothesis_ontology, max_similarity_ontology = get_most_similar_term_jacard_ngrams(term_name,ontology_mappings)
            if max_similarity_train > max_similarity_ontology:
                referent_hypothesis = referent_hypothesis_train
            elif max_similarity_ontology > max_similarity_train:
                referent_hypothesis = referent_hypothesis_ontology
            else:
                referent_hypothesis = referent_hypothesis_ontology
            if(max(max_similarity_ontology, max_similarity_train)<=0.23):
                best_similarity, best_candidate = compareTags(term_name, train_postags, ontology_postags,list(train_set_term_mappings.keys()),list(ontology_mappings.keys()))
                if (best_candidate!=None and  best_candidate in ontology_mappings.keys()):
                    referent_hypothesis = ontology_mappings[best_candidate]

        if referent_hypothesis == referent_true:
            true_count += 1
        else:
            if(term_name == 'patients with high-grade primary gastric lymphoma'):
                print()
            hypothesis_term_name = [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_hypothesis][0]
            true_term_name = [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_true][0]
            print('TermName: ' + term_name + '. Found: ' + hypothesis_term_name + '. Actual: ' + true_term_name)
    accuracy = true_count / total_count
    print(accuracy)


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


def experiment_exact_match_plus_onto_biotope_plus_jacard_average_plus_rules(train_set_term_mappings, dev_set_term_mappings, ontology_mappings):
    true_count = 0
    total_count = 0

    merged_term_name_mappings = {**train_set_term_mappings, **ontology_mappings}

    for term_name, referent_true in dev_set_term_mappings.items():
        total_count += 1
        if term_name in merged_term_name_mappings:
            referent_hypothesis = merged_term_name_mappings[term_name]
        else:
            referent_hypothesis, max_similarity, most_similar_term_name = get_most_similar_term_jacard_average(term_name, merged_term_name_mappings)

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

        if referent_hypothesis == referent_true:
                true_count += 1
        else:
            hypothesis_term_name = [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_hypothesis][0]
            true_term_name = [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_true][0]
            print('TermName: ' + term_name + '. Found: ' + hypothesis_term_name + '. Actual: ' + true_term_name)

    accuracy = true_count/total_count
    print(accuracy)


def experiment_exact_match_plus_onto_biotope_plus_plus_rules_plus_word_embedding_by_mean(train_set_term_mappings, dev_set_term_mappings, ontology_mappings):
    true_count = 0
    total_count = 0

    merged_term_name_mappings = {**train_set_term_mappings, **ontology_mappings}
    word_embedding_model = get_bio_nlp_word_vec_model()
    for term_name, referent_true in dev_set_term_mappings.items():
        total_count += 1
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
                    referent_hypothesis, max_similarity, most_similar_term_name = get_most_similar_term_word_embedding_by_mean(
                    term_name, merged_term_name_mappings, word_embedding_model)

        if referent_hypothesis == referent_true:
                true_count += 1
        else:
            hypothesis_term_name = [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_hypothesis][0]
            true_term_name = [term_name for term_name, term_id in ontology_mappings.items() if term_id == referent_true][0]
            print('TermName: ' + term_name + '. Found: ' + hypothesis_term_name + '. Actual: ' + true_term_name)

    accuracy = true_count/total_count
    print(accuracy)


