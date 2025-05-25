import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score
import libauc.losses
import libauc.optimizers
import pandas as pd

from pathlib import Path
import re
from collections import defaultdict
import json
import os

from lightning_libauc.trainer import LibAUCTrainer


class Experiment:
    def __init__(self, experiment_config):
        self.experiment_config = experiment_config
    
    def compute_metrics(self, predictions, stratify_col, always_include):
        factors = predictions[stratify_col].unique()
        metric_rows = defaultdict(list)
        for factor in factors:
            if factor in always_include:
                continue
            included_factors = [factor] + always_include
            subset = predictions[predictions[stratify_col].isin(included_factors)].reset_index()
            ap_row = dict()
            auc_row = dict()
            ap_row["factor"] = factor
            auc_row["factor"] = factor
            for model_factor in factors:
                if model_factor in always_include:
                    continue
                y_true = subset["target"]
                y_prob = subset[model_factor]
                ap_row[model_factor] = average_precision_score(y_true, y_prob)
                auc_row[model_factor] = roc_auc_score(y_true, y_prob)
            metric_rows["ap"].append(ap_row)
            metric_rows["auc"].append(auc_row)
        return pd.DataFrame(metric_rows["ap"]), pd.DataFrame(metric_rows["auc"])   


    def run(self):
        experiment_config = self.experiment_config
        
        directory = experiment_config["save_dir"]
        Path(directory).mkdir(parents=True, exist_ok=False)
        train_config = experiment_config["train_dataset"]
        train_df = pd.read_csv(train_config["file_path"])
        stratify_col = experiment_config["stratify_by"]
        factors = list(train_df[stratify_col].unique())
        always_include = experiment_config["always_include"]
        predictions = pd.DataFrame()
        for factor in factors:
            if factor in always_include:
                continue
            print(f"Training on {factor}...")
            model_name = experiment_config["model_name"]
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            loss_ = getattr(libauc.losses, experiment_config["loss_fn"])
            loss_fn = loss_(**experiment_config["loss_fn_args"])
            optimizer_ = getattr(libauc.optimizers, experiment_config["optimizer"])
            optimizer = optimizer_(model.parameters(), loss_fn=loss_fn, **experiment_config["optimizer_args"])
            included_factors = [factor] + always_include
            
            training_frame = train_df[train_df[stratify_col].isin(included_factors)].reset_index()
            train_args = experiment_config["training_args"]
            trainer = LibAUCTrainer(model, tokenizer, loss_fn, optimizer, 
                                    needs_sampler=experiment_config["needs_sampler"], 
                                    needs_index=experiment_config["needs_index"], 
                                    device=experiment_config["device"], 
                                    seed=experiment_config.get("seed",2024), 
                                    max_len=experiment_config.get("max_len", 512))
            trainer.train(training_frame, 
                            text_col=train_config["text_col"],
                            label_col=train_config["label_col"],
                            **train_args)

            test_config = experiment_config["test_dataset"]
            testing_frame = pd.read_csv(test_config["file_path"])
            test_config["text_col"]
            predicted_output = trainer.evaluate(testing_frame, test_config["text_col"], test_config["label_col"])
            predictions["target"] = predicted_output["target"].tolist()
            predictions[stratify_col] = testing_frame[stratify_col].tolist()
            predictions[str(factor)] = predicted_output["prediction"].tolist()
            ap_score = average_precision_score(predictions["target"], predictions[str(factor)])
            auc_score = roc_auc_score(predictions["target"], predictions[str(factor)])
            print("AP Score:", round(ap_score, 3))
            print("AUC Score:", round(auc_score, 3))
            
            pattern = re.compile('[\W_]+')
            cleaned_factor = pattern.sub('_', factor)
            model_save_path = os.path.join(directory, f"{model_name}_{cleaned_factor}.pt")
            trainer.save_model(model_save_path)
            print("Saved model at", model_save_path)
            print('-'*30)
        predictions.to_csv(os.path.join(directory, "predictions.csv"),index=False)
        ap_frame, auc_frame = self.compute_metrics(predictions, stratify_col, always_include)
        ap_frame.to_csv(os.path.join(directory, "ap_scores.csv"),index=False)
        auc_frame.to_csv(os.path.join(directory, "auc_scores.csv"),index=False)
        print("Saved metric computations.")
        with open(os.path.join(directory, "experiment_config.json"), 'w+') as cf:
            json.dump(experiment_config, cf, indent=4)
        print("Saved experiment config.")



    
    
    
    