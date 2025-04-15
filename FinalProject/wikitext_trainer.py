import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import argparse
import lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer

from modeling_bert import BertForMaskedLM
from modeling_electra import ElectraForMaskedLM

from electra_config import ElectraMLAConfig
from bert_config import BertMLAConfig

import os

MODEL_TOKENIZERS = {
    "electra":"google/electra-base-discriminator",
    "bert":"google-bert/bert-base-uncased",
}

class LightningWrapper(pl.LightningModule):
    def __init__(self, transformer_model, learning_rate):
        super().__init__()
        self.transformer_model = transformer_model
        self.learning_rate = learning_rate

    def training_step(self,batch):
        text_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        text_labels = batch["labels"]

        results = self.transformer_model(text_ids, attention_mask=attention_mask, labels=text_labels)

        return results.loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    
class WikiTextDataSet(Dataset):

    def __init__(self, tokenizer, wikitext_dataset_name, wikitext_dataset_subname, wikitext_split):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.huggingface_dataset = load_dataset(wikitext_dataset_name, 
                                                         name=wikitext_dataset_subname, 
                                                         split=wikitext_split)

    def __len__(self):
        return len(self.huggingface_dataset)
    
    def __getitem__(self, idx):
        sentence = self.huggingface_dataset[idx]["text"]
        batch_data = self.tokenizer(sentence, 
                                    padding="max_length",
                                    truncation=True,
                                    add_special_tokens=True)
        
        labels = torch.tensor(batch_data["input_ids"])
        attention_mask = torch.tensor(batch_data["attention_mask"])

        input_ids = labels.detach().clone()
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 2) * (input_ids != 3)
        input_ids[mask_arr] = 4

        return {"input_ids":input_ids, "attention_mask":attention_mask, "labels":labels}
        

def main():
    
    parser = argparse.ArgumentParser(prog="wikitext_trainer.py", description="Trainer script for transformer encoder models " \
                                                                             "with multi-head latent attention settings. Currently"
                                                                             "Supported models are: - ELECTRA, BERT")
    
    parser.add_argument("--dataset_name",
                        type=str,
                        required=True,
                        help="Path to hosted HuggingFace WikiText dataset or path to directory containing" \
                             "WikiText dataset you want your model of choice trained on.")
    
    parser.add_argument("--dataset_subname",
                    type=str,
                    required=True,
                    choices=["wikitext-103-v1","wikitext-2-v1"],
                    help="What subversion of wikitext you want your model trained of choice trained on.")
    
    parser.add_argument("--model_name",
                        type=str,
                        required=True,
                        choices=["electra","bert"],
                        help="Which transformer-based encoder model to train using multi-head latent attention " \
                              "on the WikiText dataset")
    
    parser.add_argument("--model_save_name",
                        type=str,
                        required=True,
                        help="Name of the model of your choice to be saved to huggingface after training is complete.")

    parser.add_argument("--model_save_directory",
                        type=str,
                        required=False,
                        default="trained_models",
                        help="Name of the directory on disk to save model of your choice after training is complete.")

    parser.add_argument("--latent_dimension",
                    type=int,
                    required=False,
                    default=0,
                    help="Size of the latent dimension the key and value attention matrices" \
                    " will be projected down to in your model of choice.")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4,
                        help="Learning rate used for controlling how fast your model of choice is pushed down its gradient.")
    
    parser.add_argument("--num_epochs",
                        type=int,
                        default=10,
                        help="Number of epochs (dataset iterations) to train your model of choice on.")
    
    parser.add_argument("--max_training_steps",
                        type=int,
                        default=10**7,
                        help="Maximum number of training steps (batches) to train your model of choice on.")

    parser.add_argument("--batch_size", 
                        type=int, 
                        default=64, 
                        help="Batch size used for training your model of choice.")
    
    parser.add_argument("--num_devices", 
                    type=int, 
                    default=1, 
                    help="Number of GPU or CPU devices to train your model of choice on.")
    
    parser.add_argument("--num_nodes", 
                    type=int, 
                    default=1, 
                    help="Number of compute nodes to train your model of choice on.")
    
    parser.add_argument("--accelerator",
                        type=str,
                        default="gpu",
                        required=False,
                        choices=["gpu","cpu"],
                        help="Use GPU or CPU devices for performing training on your model of choice.")
    
    parser.add_argument("--distributed_strategy",
                        type=str,
                        default="ddp",
                        required=False,
                        choices=["ddp","fdsp"],
                        help="Strategy for splitting datasets and model parameters across different devices during training.")

    training_arguments = parser.parse_args()
    model_name = training_arguments.model_name
    dataset_name = training_arguments.dataset_name
    dataset_subname = training_arguments.dataset_subname
    batch_size = training_arguments.batch_size

    latent_size = training_arguments.latent_dimension if training_arguments.latent_dimension != 0 else None

    num_epochs = training_arguments.num_epochs
    max_training_steps = training_arguments.max_training_steps
    num_devices = training_arguments.num_devices
    num_nodes = training_arguments.num_nodes
    accelerator = training_arguments.accelerator
    distributed_strategy = training_arguments.distributed_strategy

    learning_rate = training_arguments.learning_rate

    model_save_name = training_arguments.model_save_name
    model_save_directory = training_arguments.model_save_directory


    wikitext_train = WikiTextDataSet(tokenizer=MODEL_TOKENIZERS[model_name], 
                                        wikitext_dataset_name=dataset_name, 
                                        wikitext_dataset_subname=dataset_subname, 
                                        wikitext_split="train")
    
    wikitext_validation = WikiTextDataSet(tokenizer=MODEL_TOKENIZERS[model_name], 
                                        wikitext_dataset_name=dataset_name, 
                                        wikitext_dataset_subname=dataset_subname, 
                                        wikitext_split="validation")

    num_dataset_workers = num_devices * num_nodes * 2

    wikitext_train_dataloader = DataLoader(wikitext_train, batch_size=batch_size, num_workers=num_dataset_workers)
    wikitext_validation_dataloader = DataLoader(wikitext_validation, batch_size=batch_size, num_workers=num_dataset_workers)

    transformer_encoder_model = None

    
    if model_name == "bert":
        config = BertMLAConfig(latent_size=latent_size)
        transformer_encoder_model = BertForMaskedLM(config)
    elif model_name == "electra":
        config = ElectraMLAConfig(latent_size=latent_size)
        transformer_encoder_model = ElectraForMaskedLM(config)

    transformer_encoder_lightning = LightningWrapper(transformer_encoder_model, learning_rate)

    lightning_trainer = pl.Trainer(max_epochs=num_epochs,
                                   max_steps=max_training_steps,
                                   enable_checkpointing=True,
                                   enable_progress_bar=True,
                                   num_nodes=num_nodes,
                                   devices=num_devices,
                                   strategy=distributed_strategy,
                                   accelerator=accelerator,
                                   check_val_every_n_epoch=1)
    

    lightning_trainer.fit(transformer_encoder_lightning,
                          train_dataloaders=wikitext_train_dataloader,
                          val_dataloaders=wikitext_validation_dataloader)

    if not os.path.isdir(model_save_directory):
        os.mkdir(model_save_directory)

    transformer_encoder_model.save_pretrained(os.path.join(model_save_directory, model_save_name))
    transformer_encoder_model.push_to_hub(model_save_name)

if __name__ == "__main__":
    main()