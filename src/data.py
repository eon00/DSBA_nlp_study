from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import omegaconf
from typing import Union, List, Tuple, Literal

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, data_config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test']):
        """
        Inputs :
            data_config : omegaconf.DictConfig{
                model_name : str
                max_len : int
                valid_size : float
            }
            split : Literal['train', 'valid', 'test']
        Outputs : None
        """
        self.split = split
        self.data_config = data_config
        self.max_len = data_config.max_len
        self.tokenizer = AutoTokenizer.from_pretrained(data_config.model_name ,use_fast=True)

        
        # 데이터셋 로딩
        dataset = load_dataset('imdb')
        
        train_data = dataset['train']
        test_data = dataset['test']

        # train과 test 데이터 병합
        data_total = concatenate_datasets([train_data, test_data])

        # 데이터를 8:1:1 비율로 나누기
        train_test = data_total.train_test_split(test_size=0.1, seed=42)  # 전체 데이터를 9:1로 나누기
        train_valid = train_test['train']  # 90% 훈련 데이터
        test = train_test['test']  # 10% 테스트 데이터

        # train를 다시 8:1 씩 나누기
        train_valid = train_valid.train_test_split(test_size=1/9 ,seed=42)
        train = train_valid['train']
        valid = train_valid['test']

        if self.split == 'train':
            self.data = train
        elif self.split == 'valid':
            self.data = valid
        elif self.split == 'test':
            self.data = test
        else:
            raise ValueError("Invalid split name. Choose from ['train', 'valid', 'test'].")

        self.data = self.data.map(lambda example: self.tokenizer(example['text'], truncation=True, padding='max_length', max_length=self.max_len),batched=True)

        print(f">> SPLIT : {self.split} | Total Data Length : ", len(self.data['text']))
        
    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx) -> dict:
        """
        단일 샘플 반환 (torch.tensor 변환 포함, 기존 inputs={} 방식 유지)
        """

        input_data = {
            "input_ids": torch.tensor(self.data[idx]['input_ids']),
            "attention_mask": torch.tensor(self.data[idx]['attention_mask']),
            "label": torch.tensor(self.data[idx]['label'])
        }
        if 'token_type_ids' in self.data[idx]:
            input_data["token_type_ids"] = torch.tensor(self.data[idx]['token_type_ids'])
        return input_data
    
    @staticmethod
    def collate_fn(batch : List[Tuple[dict, int]]) -> dict:
        """
        Inputs :
            batch : List[Tuple[dict, int]]
        Outputs :
            data_dict : dict{
                input_ids : torch.Tensor
                token_type_ids : torch.Tensor
                attention_mask : torch.Tensor
                label : torch.Tensor
            }
        """
        # data_dict = {'input_ids': [], 'attention_mask': [], 'label': []}

        # # token_type_ids가 존재하는 경우 추가
        # if 'token_type_ids' in batch[0]:
        #     data_dict["token_type_ids"] = []

        # for data in batch:
        #     data_dict['input_ids'].append(data['input_ids'])
        #     data_dict['attention_mask'].append(data['attention_mask'])
        #     data_dict['label'].append(data['label'])

        #     if 'token_type_ids' in data:
        #         data_dict['token_type_ids'].append(data['token_type_ids'])

        # # torch.tensor 변환
        # data_dict['input_ids'] = torch.tensor(data_dict['input_ids'])
        # data_dict['attention_mask'] = torch.tensor(data_dict['attention_mask'])
        # data_dict['label'] = torch.tensor(data_dict['label'])

        # if 'token_type_ids' in data_dict:
        #     data_dict['token_type_ids'] = torch.tensor(data_dict['token_type_ids'])

        # import pdb; pdb.set_trace()
            
        data_dict = {
            'input_ids': torch.stack([torch.tensor(b['input_ids']) for b in batch]),
            'attention_mask': torch.stack([torch.tensor(b['attention_mask']) for b in batch]),
            'label': torch.tensor([b['label'] for b in batch])
        }

        # token_type_ids가 존재하는 경우만 추가
        if 'token_type_ids' in batch[0]:
            data_dict['token_type_ids'] = torch.stack([torch.tensor(b['token_type_ids']) for b in batch])

        # import pdb; pdb.set_trace()

        return data_dict

    
    
def get_dataloader(data_config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test']) -> torch.utils.data.DataLoader:
    """
    Output : torch.utils.data.DataLoader
    """
    config = data_config.data_config
    dataset = IMDBDataset(config, split)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=(split=='train'), pin_memory=True, collate_fn=IMDBDataset.collate_fn)
    return dataloader




