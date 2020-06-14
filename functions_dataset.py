import os
from functions_file_reading import read_lines_of_a1_file, read_lines_of_a2_file


def get_file_names(path):
    file_paths = os.listdir(path)
    file_names = []
    for file_path in file_paths:
        file_extension = file_path.split(".")[1]
        if file_extension == 'txt':
            file_name = file_path.split('.')[0]
            file_names.append(file_name)
    return file_names


def create_vocabulary(term_mappings):
    vocabulary = []
    for term_name in term_mappings:
        words = term_name.split()
        for word in words:
            if word not in vocabulary:
                vocabulary.append(word)
    return vocabulary


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













