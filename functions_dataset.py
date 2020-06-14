def create_vocabulary(term_mappings):
    vocabulary = []
    for term_name in term_mappings:
        words = term_name.split()
        for word in words:
            if word not in vocabulary:
                vocabulary.append(word)
    return vocabulary

