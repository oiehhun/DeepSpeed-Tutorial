from transformers import T5ForConditionalGeneration, AutoTokenizer

import deepspeed
import torch

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
model.load_state_dict(torch.load("/root/deepspeed_tutorial/save_path/temp/model.pt"))
model.train()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

output = model.generate(**tokenizer(['I am a boy.'], return_tensors='pt'))
print(tokenizer.batch_decode(output, skip_special_tokens=True))