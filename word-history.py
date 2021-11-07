import os
import pickle
import ilm.tokenize_util
import gdown
import time

MODEL_DIR = 'ilm/models/'

# download trained model if it's not there already
if not os.path.isfile(MODEL_DIR + 'pytorch_model.bin'):
    print('downloading...')
    url = 'https://drive.google.com/u/0/uc?id=1F8hQq8qI_P5h3NuFtZbX8GpJPMmvUWBa'
    output = MODEL_DIR + 'pytorch_model.bin'
    gdown.download(url, output, quiet=False)


tokenizer = ilm.tokenize_util.Tokenizer.GPT2
with open(os.path.join(MODEL_DIR, 'additional_ids_to_tokens.pkl'), 'rb') as f:
    additional_ids_to_tokens = pickle.load(f)
additional_tokens_to_ids = {v:k for k, v in additional_ids_to_tokens.items()}
try:
    ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, tokenizer)
except ValueError:
    print('Already updated')

import torch
from transformers import GPT2LMHeadModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
model.eval()
_ = model.to(device)

# Create context


# Using readlines()
file = open('input.txt', 'r')
lines = file.readlines()

#while(True):
fullText = ""
context_input = ""
textLine = ""
#context_input = input("Enter the message: ")
for line in lines:
    textLine = line.strip()
    print(textLine)

    context_input = ""
    context_input = fullText + textLine

    blanks_number = context_input.count('_')

    context = context_input.strip()

    context_ids = ilm.tokenize_util.encode(context, tokenizer)

    # Replace blanks with appropriate tokens from left to right
    _blank_id = ilm.tokenize_util.encode(' _', tokenizer)[0]
    for i in range(0, blanks_number):
        context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_word|>']

    from ilm.infer import infill_with_ilm

    generated = infill_with_ilm(
        model,
        additional_tokens_to_ids,
        context_ids,
        num_infills=1)

    fullText = ilm.tokenize_util.decode(generated[0], tokenizer)

    time.sleep(0.1)

file_object = open('result-word-history.txt', 'w')
file_object.write(fullText)
file_object.close()