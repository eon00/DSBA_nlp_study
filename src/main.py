import wandb 
from tqdm import tqdm
import os

import argparse
import torch
import torch.nn
import omegaconf
from omegaconf import OmegaConf

from utils import load_config
from model import EncoderForClassification
from data import get_dataloader

import logging

from transformers import set_seed, get_scheduler
set_seed(42)

def train_iter(model, inputs, optimizer, device, scheduler):
    inputs = {key : value.to(device) for key, value in inputs.items()}
    loss, _ = model(**inputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()

def valid_iter(model, inputs, device):
    inputs = {key : value.to(device) for key, value in inputs.items()}
    loss, logits = model(**inputs)
    accuracy = calculate_accuracy(logits, inputs['label'])    
    return loss.item(), accuracy

def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)

def main(configs : omegaconf.DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # load model
    model = EncoderForClassification(configs).to(device)
    
    # load data
    train_dataloader = get_dataloader(configs, split="train")
    valid_dataloader = get_dataloader(configs, split="valid")
    test_dataloader = get_dataloader(configs, split="test")
    
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
    scheduler = get_scheduler(name="constant", optimizer=optimizer)
    
    wandb.init(project=configs.train_config.wandb_project, name=configs.train_config.run_name)
    best_valid_accuracy = 0.0
    
    for epoch in range(configs.train_config.epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            loss = train_iter(model, batch, optimizer, device, scheduler)
            total_train_loss += loss
            # print("total_train_loss:", total_train_loss)
            # print("len(train_dataloader): ", len(train_dataloader))
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        model.eval()
        total_valid_loss = 0
        total_valid_accuracy = 0
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc=f"Validation Epoch {epoch+1}"):
                loss, accuracy = valid_iter(model, batch, device)
                total_valid_loss += loss
                # print("total_valid_loss:", total_valid_loss)
                # print("len(valid_dataloader): ", len(valid_dataloader))
                total_valid_accuracy += accuracy
        
        avg_valid_loss = total_valid_loss / len(valid_dataloader)
        avg_valid_accuracy = total_valid_accuracy / len(valid_dataloader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {avg_valid_accuracy:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss, "valid_loss": avg_valid_loss, "valid_acc": avg_valid_accuracy})
        
        if avg_valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = avg_valid_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Checkpoint saved at epoch {epoch+1} with accuracy {best_valid_accuracy:.4f}")
    
    print("Starting final test evaluation...")
    model.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Final Test"):
            loss, accuracy = valid_iter(model, batch, device)
            total_test_loss += loss
            total_test_accuracy += accuracy
    
    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_accuracy = total_test_accuracy / len(test_dataloader)
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")
    wandb.log({"test_loss": avg_test_loss, "test_acc": avg_test_accuracy})
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    args = parser.parse_args()
    configs = load_config(args.config)
    main(configs)
