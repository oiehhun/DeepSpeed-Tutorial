# https://medium.com/@roysr/pytorch-custom-dataloader-and-microsoft-deepspeed-d26dcf03b212
# https://junbuml.ee/transformers-deepspeed-new-bert-model

from transformers import T5ForConditionalGeneration, AutoTokenizer

import deepspeed
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

import torch

import argparse
import json

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, input_sent_list, target_sent_list, tokenizer, max_length=512):
        super().__init__()
        self.input_sent_list = input_sent_list
        self.target_sent_list = target_sent_list
        self.tokenizer = tokenizer
        
        self.input_batch_encoding = self.tokenizer(input_sent_list, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        self.target_batch_encoding = self.tokenizer(target_sent_list, max_length=max_length, padding=True, truncation=True, return_tensors='pt')

        self.length = len(input_sent_list)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        input_input_ids = self.input_batch_encoding.input_ids[idx]
        input_attention_mask = self.input_batch_encoding.attention_mask[idx]
        target_input_ids = self.target_batch_encoding.input_ids[idx]
        return input_input_ids, input_attention_mask, target_input_ids


def main():
    
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepspeed_config', type=str)
    parser.add_argument('--local_rank', type=int, default=-1)
    
    #parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

    #with open(args.config, "r") as config_str:
    #    deepspeed_config = json.load(config_str)
        
    
    #mini dataset
    input_sent_list = ['I am a boy.']*128
    target_sent_list = ['You are a girl.']*128

    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    model.train()
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    
    my_torch_dataset = BasicDataset(input_sent_list, target_sent_list, tokenizer)
    my_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   
    my_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=my_optimizer,
                                            lr_lambda=lambda epoch: 0.95 ** epoch,
                                            last_epoch=-1,
                                            verbose=False)
    
    #https://deepspeed.readthedocs.io/en/latest/initialize.html
    model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(model=model, 
                                                                                    model_parameters=model.parameters(),
                                                                                    training_data = my_torch_dataset,
                                                                                    optimizer = my_optimizer,
                                                                                    lr_scheduler = my_scheduler,
                                                                                    args=cmd_args)

    '''
    model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(model=model, 
                                                                                    model_parameters=model.parameters(),
                                                                                     optimizer = my_optimizer,
                                                                                    training_data = my_torch_dataset,
                                                                                    args=cmd_args)
     '''
     
    count = 0
    for epoch in range(10):
        for idx, (input_ids, attention_mask, y) in enumerate(training_dataloader):
            #if model_engine.local_rank == 0:
            #    print(count)
            #    count+=1
            
            input_ids = input_ids.to(model_engine.local_rank)
            attention_mask = attention_mask.to(model_engine.local_rank)
            y = y.to(model_engine.local_rank)
            
            #forward
            loss = model_engine(input_ids = input_ids, attention_mask=attention_mask, labels=y).loss
            
            #backprop
            
            model_engine.backward(loss)
            
            #weight update
            model_engine.step()
        
        
        if model_engine.local_rank == 0:
            print("LOSS " + str(loss.item()))
            #ckpt_id = loss.item()
            #model_engine.save_checkpoint('./save_path', ckpt_id)
    
    model_engine.save_checkpoint('./save_path/temp')
            
if __name__ == '__main__':
    main()