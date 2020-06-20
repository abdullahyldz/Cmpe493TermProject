import spacy
from pronto import Ontology
from functions_dataset import create_vocabulary


def find_headword(term, nlp):

    doc = nlp(term)
    #print(term)
    compound = ''
    headword = ''
    for token in doc:
        #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
         #     token.shape_, token.is_alpha, token.is_stop)
        if token.dep_ == 'ROOT':
            headword = token.text
        elif token.dep_ == 'compound': #bazı durumlarda compound da yararlı olabilir
            compound = token.text + ' '

    return headword


if __name__ == '__main__':
    ontology = Ontology('./Datasets/OntoBiotope.obo')
    ontology_mapping = read_ontology_mapping(path='./Datasets/OntoBiotope.obo')

    train_set_term_mapping = create_term_mapping(dataset_directory='./Datasets/train/')
    dev_set_term_mapping = create_term_mapping(dataset_directory='./Datasets/dev/')
    for term, reference in dev_set_term_mapping.items():
        find_headword(term)
