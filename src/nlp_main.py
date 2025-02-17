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
    loss, logits = model(**inputs)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    # Accuracy 계산 추가
    accuracy = calculate_accuracy(logits, inputs['label'])
    return loss.item(), accuracy

def valid_iter(model, inputs, device):
    inputs = {key : value.to(device) for key, value in inputs.items()}
    loss, logits = model(**inputs)
    accuracy = calculate_accuracy(logits, inputs['label'])    
    
    # 오분류된 샘플 저장
    incorrect_preds = (logits.argmax(dim=-1) != inputs["label"])  
    incorrect_samples = [inputs["input_ids"][i].tolist() for i in range(len(incorrect_preds)) if incorrect_preds[i]]  
    
    return loss.item(), accuracy, incorrect_samples

def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)

def train_model(configs : omegaconf.DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda:0":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model = EncoderForClassification(configs).to(device)
    
    # Load data
    train_dataloader = get_dataloader(configs, split="train")
    valid_dataloader = get_dataloader(configs, split="valid")
    test_dataloader = get_dataloader(configs, split="test")
    
    # Set optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scheduler = get_scheduler(name="constant", optimizer=optimizer)
    
    wandb.init(project=configs.train_config.wandb_project, name=configs.train_config.run_name)
    best_valid_accuracy = 0.0
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(configs.train_config.epochs):
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0
        # import pdb; pdb.set_trace()
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            # import pdb; pdb.set_trace()

            loss, accuracy = train_iter(model, batch, optimizer, device, scheduler)
            total_train_loss += loss
            total_train_accuracy += accuracy
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        
        model.eval()
        total_valid_loss = 0
        total_valid_accuracy = 0
        all_incorrect_samples = []
        
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc=f"Validation Epoch {epoch+1}"):
                loss, accuracy, incorrect_samples = valid_iter(model, batch, device)
                total_valid_loss += loss
                total_valid_accuracy += accuracy
                all_incorrect_samples.extend(incorrect_samples)
        
        avg_valid_loss = total_valid_loss / len(valid_dataloader)
        avg_valid_accuracy = total_valid_accuracy / len(valid_dataloader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}, Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {avg_valid_accuracy:.4f}")
        
        # Logging to wandb
        wandb.log({
            "epoch": epoch+1, 
            "train_loss": avg_train_loss, 
            "train_acc": avg_train_accuracy,
            "valid_loss": avg_valid_loss, 
            "valid_acc": avg_valid_accuracy,
            "incorrect_samples": all_incorrect_samples,
            "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        })
        
        # Checkpoint 저장
        if avg_valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = avg_valid_accuracy
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path} with accuracy {best_valid_accuracy:.4f}")
    
    # print("Starting final test evaluation...")
    # model.eval()
    # total_test_loss = 0
    # total_test_accuracy = 0
    
    # with torch.no_grad():
    #     for batch in tqdm(test_dataloader, desc="Final Test"):
    #         loss, accuracy, _ = valid_iter(model, batch, device)
    #         total_test_loss += loss
    #         total_test_accuracy += accuracy
    
    # avg_test_loss = total_test_loss / len(test_dataloader)
    # avg_test_accuracy = total_test_accuracy / len(test_dataloader)
    
    # print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")
    # wandb.log({"test_loss": avg_test_loss, "test_acc": avg_test_accuracy})
    
    # wandb.finish()
    
    
def evaluate_model(configs): ## 테스트용 모델 코드 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = configs.train_config.checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = EncoderForClassification(configs).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    test_dataloader = get_dataloader(configs, split="test")
    total_test_loss = 0
    total_test_accuracy = 0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating Model"):
            inputs = {key: value.to(device) for key, value in batch.items()}
            loss, logits = model(**inputs)
            accuracy = calculate_accuracy(logits, inputs['label'])
            total_test_loss += loss.item()
            total_test_accuracy += accuracy
    
    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_accuracy = total_test_accuracy / len(test_dataloader)
    
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")
    
    wandb.init(project=configs.train_config.wandb_project, name="evaluation")
    wandb.log({"test_loss": avg_test_loss, "test_acc": avg_test_accuracy})
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training & Evaluation Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="Mode: train or test")
    args = parser.parse_args()
    
    configs = load_config(args.config)
    
    if args.mode == "train":
        train_model(configs)
    elif args.mode == "test":
        evaluate_model(configs)
    elif args.mode == "all":
        train_model(configs)
        evaluate_model(configs)
    else: 
        raise "not entering the argument(mode) for main.py"


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Training Script")
#     parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
#     args = parser.parse_args()
#     configs = load_config(args.config)
#     main(configs)
