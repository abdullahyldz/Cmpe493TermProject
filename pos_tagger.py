
import nltk
from functions_text_processing import jaccard_similarity, jaccard_similarity_stemming

def compareTags(term_name, postag_train_set, postag_ontology_set, train_set_term_mappings, ontology_mappings):
    if(term_name == 'patients with chronic active gastritis'):
        print('')
    best_similarity = 0
    best_candidate = None
    term_postag = list(extract_NN(term_name))
    most_similar_term_id = None

    for index, postag in enumerate(postag_train_set):
        possibility = jaccard_similarity(term_postag, postag)
        if (possibility > best_similarity):
            best_similarity = possibility
            best_candidate = train_set_term_mappings[index]

    for index, train_set_term_map in enumerate(postag_ontology_set):
        possibility = jaccard_similarity_stemming(term_postag, postag)
        if (possibility > best_similarity):
            best_similarity = possibility
            best_candidate = ontology_mappings[index]

    return best_similarity, best_candidate

def createTags(train_set_term_mappings, ontology_mappings):
    best_similarity = 0
    best_candidate = None
    train_set_postags = [list(extract_NN(train_key)) for train_key in train_set_term_mappings.keys()]
    ontology_set_postags = [list(extract_NN(ontology_key)) for ontology_key in ontology_mappings.keys()]

    return train_set_postags, ontology_set_postags

def extract_NN(sent):
    grammar = r"""
    NBAR:
        # Nouns and Adjectives, terminated with Nouns
        {<NN.*>*<NN.*>}

    NP:
        {<NBAR>}
        # Above, connected with in/of/etc...
        {<NBAR><IN><NBAR>}
    """
    chunker = nltk.RegexpParser(grammar)
    ne = set()
    chunk = chunker.parse(nltk.pos_tag(nltk.word_tokenize(sent)))
    for tree in chunk.subtrees(filter=lambda t: t.label() == 'NP'):
        ne.add(' '.join([child[0] for child in tree.leaves()]))
    return ne
