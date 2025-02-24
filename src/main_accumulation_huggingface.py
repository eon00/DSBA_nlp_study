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

from transformers import set_seed, get_scheduler
from accelerate import Accelerator

torch.cuda.empty_cache()
set_seed(42)

def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)

def train_iter(model, inputs, optimizer, accelerator, accum_step):
    inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}
    
    with accelerator.autocast():  # Mixed Precision 지원
        loss, _ = model(**inputs)
    
    # Gradient Accumulation을 고려한 Loss Scaling
    loss = loss / accum_step  
    accelerator.backward(loss)
    
    if accelerator.sync_gradients:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    return loss.item()

def valid_iter(model, inputs, accelerator):
    inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}
    with torch.no_grad():
        loss, logits = model(**inputs)
    accuracy = calculate_accuracy(logits, inputs['label'])
    return loss.item(), accuracy

def main(configs: omegaconf.DictConfig):
    accelerator = Accelerator(gradient_accumulation_steps=configs.train_config.get("gradient_accumulation_steps", 1))
    device = accelerator.device

    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
    
    model = EncoderForClassification(configs).to(device)
    train_dataloader = get_dataloader(configs, split="train")
    valid_dataloader = get_dataloader(configs, split="valid")
    test_dataloader = get_dataloader(configs, split="test")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.train_config.get("learning_rate", 1e-5))
    scheduler = get_scheduler(name="constant", optimizer=optimizer)
    accum_step = configs.train_config.get("gradient_accumulation_steps", 1)
    
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    
    wandb.init(project=configs.train_config.wandb_project, name=configs.train_config.run_name)
    best_valid_accuracy = 0.0
    
    for epoch in range(configs.train_config.epochs):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            loss = train_iter(model, batch, optimizer, accelerator, accum_step)
            total_train_loss += loss
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        model.eval()
        total_valid_loss = 0
        total_valid_accuracy = 0
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc=f"Validation Epoch {epoch+1}"):
                loss, accuracy = valid_iter(model, batch, accelerator)
                total_valid_loss += loss
                total_valid_accuracy += accuracy
        
        avg_valid_loss = total_valid_loss / len(valid_dataloader)
        avg_valid_accuracy = total_valid_accuracy / len(valid_dataloader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {avg_valid_accuracy:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss, "valid_loss": avg_valid_loss, "valid_acc": avg_valid_accuracy})
        
        if avg_valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = avg_valid_accuracy
            accelerator.save_model(model, "./accumulation/best_model.pth")
            print(f"Checkpoint saved at epoch {epoch+1} with accuracy {best_valid_accuracy:.4f}")
        
        scheduler.step()  # 학습 후에 업데이트
    
    print("Starting final test evaluation...")
    model.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Final Test"):
            loss, accuracy = valid_iter(model, batch, accelerator)
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