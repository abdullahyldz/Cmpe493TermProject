import os
from collections import defaultdict
from functions_search import search_exact_match_plus_onto_biotope, search_exact_match_plus_onto_biotope_plus_jacard_average


def create_term_mapping_from_a1_and_a2_lines(a1_lines, a2_lines):
    term_mapping = dict()
    for term_index, term_name in a1_lines.items():
        term_mapping[term_name] = a2_lines[term_index]
    return term_mapping


def create_term_mapping(dataset_directory):
    file_names = get_file_names(dataset_directory)
    term_mappings = dict()

    for file_name in file_names:
        try:
            a1_file_path = dataset_directory + file_name + '.a1'
            a1_lines = read_lines_of_a1_file(a1_file_path)

            a2_file_path = dataset_directory + file_name + '.a2'
            a2_lines = read_lines_of_a2_file(a2_file_path)

            term_mappping_in_file = create_term_mapping_from_a1_and_a2_lines(a1_lines, a2_lines)
            term_mappings.update(term_mappping_in_file)

        except Exception:
            # 2 dosyada beta gibi yunan harfleri encoding hatası veriyor, onları şimdilik eklemiyorum.
            print(Exception)

    return term_mappings


def get_file_names(path):
    file_paths = os.listdir(path)
    file_names = []
    for file_path in file_paths:
        file_extension = file_path.split(".")[1]
        if file_extension == 'txt':
            file_name = file_path.split('.')[0]
            file_names.append(file_name)
    return file_names


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


def parse_a1_files(dataset_directory):
    a1_data = defaultdict(dict)

    file_paths = []
    for file in os.listdir(dataset_directory):
        if file.endswith(".a1"):
            file_paths.append(os.path.join(dataset_directory, file))

    for file_path in file_paths:
        a1_lines = read_lines_of_a1_file(file_path)
        file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
        for term_number, term_name in a1_lines.items():
            a1_data[file_name_without_extension][term_number] = term_name

    return a1_data


def write_a2_files(dataset_directory, a1_data, train_set_term_mapping, ontology_mapping):
    for file_name, term_dictionary in a1_data.items():
        output_file = open(os.path.join(dataset_directory, file_name + '.a2'), mode="a", encoding='utf8')
        NER_index = 1
        for term_number, term_name in term_dictionary.items():
            output_file.write('N' + str(NER_index) + '\t' + 'OntoBiotope ' + 'Annotation:' + term_number + ' Referent:' + search_exact_match_plus_onto_biotope_plus_jacard_average(train_set_term_mapping, ontology_mapping, term_name) + '\n')
            NER_index += 1
        output_file.close()

