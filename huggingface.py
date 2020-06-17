import torch
from transformers import BertModel, AutoTokenizer, AutoModel

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")

# Load pre-trained model (weights)
model = AutoModel.from_pretrained("monologg/biobert_v1.0_pubmed_pmc")

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()



input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
print(last_hidden_states)


input_sentence = torch.tensor(tokenizer.encode("[CLS] My sentence")).unsqueeze(0)
out = model(input_sentence)
embeddings_of_last_layer = out[0]
cls_embeddings = embeddings_of_last_layer[0]
print(cls_embeddings)

