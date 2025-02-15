from transformers import T5ForConditionalGeneration, AutoTokenizer

from tqdm import tqdm

import os
import deepspeed
import torch

local_rank = int(os.getenv('LOCAL_RANK', '0'))
device = f'cuda:{local_rank}'

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
model.load_state_dict(torch.load("/root/deepspeed_tutorial/save_path/temp/model.pt"))
zero_config = {
    "stage": 3,
    "contiguous_gradients": True,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 1e7,
    "stage3_param_persistence_threshold": 1e5,
    "reduce_bucket_size": 1e7,
    "sub_group_size": 1e9
    }
#Tensor Parallelism
ds_engine = deepspeed.init_inference(model,
                                 mp_size=8,
                                 dtype=torch.float)


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

input_list = ['I am a boy.']*8


for input_sent in tqdm(input_list):

    output = ds_engine.generate(**tokenizer([input_sent], max_length=2048, padding=True, truncation=True, return_tensors='pt').to('cuda'), max_length=4096)

    if local_rank== 0:
        print(tokenizer.batch_decode(output, skip_special_tokens=True))