def read_lines_of_a1_file(path):
    file = open(path, mode='r', encoding='utf8')
    a1_lines = dict()
    while True:
        line = file.readline().rstrip()
        if not line:
            break
        line_parts = line.split('\t')
        term_index = line_parts[0]
        term_type_and_positions = line_parts[1]
        term_name = line_parts[2]
        if 'Habitat' in term_type_and_positions:
            a1_lines[term_index] = term_name
    return a1_lines


def read_lines_of_a2_file(path):
    file = open(path, mode='r', encoding='utf8')
    a2_lines = dict()
    while True:
        line = file.readline().rstrip()
        if not line:
            break
        line_second_part = line.split('\t')[1]
        parts = line_second_part.split(' ')
        if parts[0] == 'OntoBiotope':
            term_index = parts[1].split(':')[1]
            referent_parts = parts[2].split(':')
            OBT_index = referent_parts[1]+':'+referent_parts[2]
            a2_lines[term_index] = OBT_index
    return a2_lines


def read_lines_of_file(path):
    with open(path, mode='r', encoding='utf8') as file:
        lines = file.read().splitlines()
        return lines


def read_ontology_mapping(path):
    ontology_lines = read_lines_of_file(path)
    ontology_mapping = dict()
    for line_index in range(len(ontology_lines)):
        line = ontology_lines[line_index]
        if line == '[Term]':
            id_line_index = line_index + 1
            name_line_index = line_index + 2
            id = ontology_lines[id_line_index][4:]
            name = ontology_lines[name_line_index][6:]
            ontology_mapping[name] = id
    return ontology_mapping


def read_acronyms_file(path):
    acronym_lines = read_lines_of_file(path)
    acronyms = dict()
    for acronym_line in acronym_lines:
        temp = acronym_line.split(':')
        abbreviation = temp[0][:-1]
        definition = temp[1][1:]
        acronyms[abbreviation] = definition
    return acronyms
