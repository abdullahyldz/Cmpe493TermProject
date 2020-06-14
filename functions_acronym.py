from functions_file_io import read_lines_of_file

def normalize_abbreviations(normalized_term, acronyms):
    term_words = normalized_term.split()
    normalized_words = []
    for term_word in term_words:
        if term_word.isupper():
            definition = find_definition_for_abbreviation(term_word, acronyms)
            definition_words = definition.split()
            for definition_word in definition_words:
                normalized_words.append(definition_word)
        else:
            normalized_words.append(term_word)
    normalized_term = ' '.join(normalized_words)
    return normalized_term


def find_definition_for_abbreviation(abbreviation, acronyms):
    if abbreviation in acronyms:
        return acronyms[abbreviation]
    else:
        return abbreviation
