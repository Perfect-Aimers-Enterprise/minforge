from mindforge_ml.datasets.loader import seq2seqdataset
from mindforge_ml.utils import tokenize, smart_tokenizer, ml_vocab_size, pad_token_id
from mindforge_ml.visualization import plot_losses, plot_accuracy
from mindforge_ml.supervised.model import MFTransformerSeq2Seq
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

vocab_size = ml_vocab_size()
data = seq2seqdataset()


text = "Hello, how are you doing?"
# token = tokenize(text)
# print(token)

all_input_ids = []
all_attention_mask = []
all_labels = []

for item in data:
    query = item["question"]
    target = item["answer"]

    input_ids, attention_mask, labels = smart_tokenizer(query, target)

    all_input_ids.append(input_ids)
    all_attention_mask.append(attention_mask)
    all_labels.append(labels)

# Stack into tensors
input_ids = torch.cat(all_input_ids)
attention_mask = torch.cat(all_attention_mask)
labels = torch.cat(all_labels)

print(f"{input_ids}, {attention_mask}, {labels}")
model = MFTransformerSeq2Seq(vocab_size)
logits = model.fit(input_ids, attention_mask, labels)

model.predict('My baby cries a lot')
