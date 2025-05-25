import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from libauc.sampler import DualSampler
from libauc.utils import set_all_seeds

from lightning_libauc.textdataset import TextDataset
import torch

from tqdm import tqdm

class LibAUCTrainer:
    def __init__(self, model, tokenizer, loss_fn, optimizer, needs_sampler=False, needs_index=False, max_len=512, device=None, seed=2024):
        self.model = model
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.max_len = max_len 
        self.seed = seed
        self.needs_index = needs_index
        self.needs_sampler=needs_sampler
        set_all_seeds(seed)
  
    def make_dataloader(self, dataframe,  batch_size, text_col, label_col, shuffle=False, use_sampler=False, sampling_rate=0.5):
        # make dataset
        dataset = TextDataset(dataframe, text_col, label_col)        
        # make dualsampler
        if use_sampler:
            sampler = DualSampler(dataset, batch_size=batch_size, sampling_rate=sampling_rate, random_seed=self.seed)
            loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    
    def train(self, training_df, epochs, batch_size, text_col, label_col, shuffle=False, sampling_rate=0.5, apply_sigmoid=True):
       
        training_loader = self.make_dataloader(training_df, batch_size, text_col, label_col, shuffle, use_sampler=self.needs_sampler,sampling_rate=sampling_rate )

        self.model.train()
        for _ in range(epochs):
            for _,data in enumerate(tqdm(training_loader)):
                texts, targets, indices = data
                inputs = self.tokenizer.batch_encode_plus(
                    texts,
                    max_length=self.max_len,
                    add_special_tokens=True,
                    padding="max_length",
                    return_token_type_ids=True,
                    truncation=True,
                    return_tensors='pt'
                )
                ids = inputs['input_ids'].to(self.device)
                mask = inputs['attention_mask'].to(self.device)
                
                indices = indices.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()     
                outputs = self.model(ids, attention_mask=mask)
           
                if apply_sigmoid:
                    pred = torch.sigmoid(outputs.logits)
                else:
                    pred = outputs.logits
                pred = torch.flatten(pred)
                if not self.needs_index:
                    loss = self.loss_fn(pred, targets)
                else:
                    loss = self.loss_fn(pred, targets, indices)
                loss.backward()
                self.optimizer.step()
        return

    def evaluate(self, testing_df, text_col, label_col, batch_size=1):
        testing_loader = self.make_dataloader(testing_df, batch_size, text_col, label_col)
        predictions = list()
        true_labels = list()
        self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(tqdm(testing_loader)):
                texts, targets, _ = data
                inputs = self.tokenizer.batch_encode_plus(
                    texts,
                    max_length=self.max_len,
                    add_special_tokens=True,
                    padding="max_length",
                    return_token_type_ids=True,
                    truncation=True,
                    return_tensors='pt'
                )
                ids = inputs['input_ids'].to(self.device)
                mask = inputs['attention_mask'].to(self.device)
                outputs = self.model(ids, attention_mask=mask)
                #logits = torch.softmax(logits, dim=1)
                logits = torch.sigmoid(outputs.logits)
                predictions.append(logits.cpu().detach().numpy())
                true_labels.append(targets.cpu().numpy())
            predictions = np.concatenate(predictions)
            true_labels = np.concatenate(true_labels)
        output = pd.DataFrame()
        output["target"] = true_labels
        output["prediction"] = predictions
        return output
    

    def save_model(self, model_path):
        torch.save(self.model, model_path)

    def predict(self, texts):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer.batch_encode_plus(
                texts,
                max_length=self.max_len,
                add_special_tokens=True,
                padding="max_length",
                return_token_type_ids=True,
                truncation=True,
                return_tensors='pt'
            )
            ids = inputs['input_ids'].to(self.device)
            mask = inputs['attention_mask'].to(self.device)
            outputs = self.model(ids, attention_mask=mask)
            #logits = torch.softmax(logits, dim=1)
            logits = torch.sigmoid(outputs)
            return logits.cpu().detach().numpy().flatten()
            
        