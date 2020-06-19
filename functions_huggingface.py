import torch
from transformers import AutoTokenizer, AutoModel
# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
from sklearn.metrics.pairwise import cosine_similarity
import operator
import numpy as np

logging.basicConfig(level=logging.INFO)


def get_bert_model_and_tokenizer():
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")

    # Load pre-trained model (weights)
    model = AutoModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.eval()

    return model, tokenizer


def write_bert_term_name_embeddings(output_path, term_mapping, bert_model, bert_tokenizer):
    output_file = open(output_path, mode='w', encoding='utf8')

    for term_name in term_mapping:
        input_token_ids = torch.tensor(bert_tokenizer.encode(term_name)).unsqueeze(0)  # Batch size 1
        outputs = bert_model(input_token_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        sentence_embedding = last_hidden_states[0][0]
        sentence_embedding = sentence_embedding.cpu().detach().numpy()
        embedding_part = ''
        for embedding_dim in sentence_embedding:
            embedding_part = embedding_part + str(embedding_dim) + ' '
        line = term_name + '\t' + embedding_part + '\n'
        output_file.write(line)
    output_file.close()


def read_bert_term_name_embeddings(input_path):
    input_file = open(input_path, mode='r', encoding='utf8')
    term_name_embeddings = dict()

    while True:
        line = input_file.readline().rstrip()
        if not line:
            break
        term_name = line.split('\t')[0]
        term_name_embedding_string = line.split('\t')[1]
        term_name_embedding = np.array([float(i) for i in term_name_embedding_string.split(' ')])
        term_name_embeddings[term_name] = term_name_embedding
    input_file.close()

    return term_name_embeddings


def get_term_name_with_max_similarity(term_name_similarities):
    most_similar_term_name = max(term_name_similarities.items(), key=operator.itemgetter(1))[0]
    max_similarity = term_name_similarities[most_similar_term_name]
    return most_similar_term_name, max_similarity


def get_most_similar_term_biobert(term_name, term_mappings, term_name_embeddings, bert_model, bert_tokenizer):
    input_token_ids = torch.tensor(bert_tokenizer.encode(term_name)).unsqueeze(0)  # Batch size 1
    outputs = bert_model(input_token_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    current_term_name_embedding = last_hidden_states[0][0].cpu().detach().numpy()
    term_name_similarities = dict()
    for term_name, term_name_embedding in term_name_embeddings.items():
        cos_sim = cosine_similarity(current_term_name_embedding.reshape(1, -1), term_name_embedding.reshape(1,-1))[0][0]
        term_name_similarities[term_name] = cos_sim

    most_similar_term_name, max_similarity = get_term_name_with_max_similarity(term_name_similarities)
    most_similar_term_id = term_mappings[most_similar_term_name]

    return most_similar_term_id, max_similarity


# bert_model, bert_tokenizer = get_bert_model_and_tokenizer()
# write_bert_term_name_embeddings('./Datasets/train_set_embeddings.txt', train_set_term_mapping, bert_model, bert_tokenizer)
# write_bert_term_name_embeddings('./Datasets/ontology_embeddings.txt', ontology_mapping, bert_model, bert_tokenizer)
