from pronto import Ontology
from functions_dataset import create_vocabulary
from functions_file_io import read_ontology_mapping, read_acronyms_file, create_term_mapping, parse_a1_files, \
    write_a2_files
from functions_experiment import experiment_exact_match, experiment_exact_match_plus_onto_biotope, \
    experiment_exact_match_plus_onto_biotope_plus_tfidf, \
    experiment_exact_match_plus_onto_biotope_plus_tfidf_plus_onto_tfidf, \
    experiment_exact_match_plus_onto_biotope_plus_jacard_ngrams, \
    experiment_exact_match_plus_onto_biotope_plus_jacard_tokens, \
    experiment_exact_match_plus_onto_biotope_plus_jacard_average, \
    experiment_exact_match_plus_onto_biotope_plus_jacard_average_plus_rules

from functions_huggingface import read_bert_term_name_embeddings, get_bert_model_and_tokenizer
import sys

if __name__ == "__main__":
    # acronym_mapping = read_acronyms_file(path='./Datasets/acronym_set.txt')

    # ontology = Ontology('./Datasets/OntoBiotope.obo')
    ontology_mapping = read_ontology_mapping(path='./Datasets/OntoBiotope.obo')

    train_set_term_mapping = create_term_mapping(dataset_directory='./Datasets/train/')
    dev_set_term_mapping = create_term_mapping(dataset_directory='./Datasets/dev/')

    # vocabulary = create_vocabulary(train_set_term_mapping)

    # bert_model, bert_tokenizer = get_bert_model_and_tokenizer()
    # train_embeddings = read_bert_term_name_embeddings('./Datasets/train_set_embeddings.txt')
    # ontology_embeddings = read_bert_term_name_embeddings('./Datasets/ontology_embeddings.txt')


    # experiments
    # experiment_exact_match(train_set_term_mapping, dev_set_term_mapping)
    # experiment_exact_match_plus_onto_biotope(train_set_term_mapping, dev_set_term_mapping, ontology_mapping)

    # experiment_exact_match_plus_onto_biotope_plus_tfidf(train_set_term_mapping, dev_set_term_mapping, ontology_mapping)

    # experiment_exact_match_plus_onto_biotope_plus_tfidf_plus_ontology_tfidf(train_set_term_mapping, dev_set_term_mapping, ontology_mapping)

    # experiment_exact_match_plus_onto_biotope_plus_biobert(train_set_term_mapping, dev_set_term_mapping, ontology_mapping, train_embeddings, ontology_embeddings, bert_model, bert_tokenizer)

    # experiment_exact_match_plus_onto_biotope_plus_jacard_ngrams(train_set_term_mapping, dev_set_term_mapping, ontology_mapping)

    # experiment_exact_match_plus_onto_biotope_plus_jacard_tokens(train_set_term_mapping, dev_set_term_mapping, ontology_mapping)

    # experiment_exact_match_plus_onto_biotope_plus_jacard_average(train_set_term_mapping, dev_set_term_mapping, ontology_mapping)

    experiment_exact_match_plus_onto_biotope_plus_jacard_average_plus_rules(train_set_term_mapping, dev_set_term_mapping, ontology_mapping)


    # official evaluation tool
    # test_set_a1_data = parse_a1_files(dataset_directory='./Datasets/test/')
    # write_a2_files(dataset_directory='./Datasets/test/', a1_data=test_set_a1_data, train_set_term_mapping=train_set_term_mapping, ontology_mapping=ontology_mapping)
    sys.exit(0)
