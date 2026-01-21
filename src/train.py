import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import *
from dataset import PokemonDataset
from model import PokemonCNN
from transforms import train_transform, val_transform
from utils import prepare_dataframe

# train cnn
def train_epoch(model, loader, optimizer, type_criterion, stat_criterion, device):
    model.train()
    total_loss = 0
    
    for imgs, types, stats in loader:
        imgs = imgs.to(device)
        types = types.to(device)
        stats = stats.to(device)
        
        optimizer.zero_grad()
        pred_types, pred_stats = model(imgs)
        
        loss_type = type_criterion(pred_types, types)
        loss_stat = stat_criterion(pred_stats, stats)
        loss = loss_type + STAT_LOSS_WEIGHT * loss_stat
        
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

# run validation
def validate(model, loader, type_criterion, stat_criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for imgs, types, stats in loader:
            imgs = imgs.to(device)
            types = types.to(device)
            stats = stats.to(device)
            
            pred_types, pred_stats = model(imgs)
            
            loss_type = type_criterion(pred_types, types)
            loss_stat = stat_criterion(pred_stats, stats)
            loss = loss_type + STAT_LOSS_WEIGHT * loss_stat
            total_loss += loss.item()
    
    return total_loss / len(loader)

# train and create the cnn 
def main():
    print(f"Starting training using {DEVICE}")
    
    df = pd.read_csv(DATA_CSV)
    df = prepare_dataframe(df)
    print(f"Total Sample Count: {len(df)}")
    
    train_df, val_df = train_test_split(df, test_size=1-TRAIN_SPLIT, random_state=42)
    train_dataset = PokemonDataset(train_df, IMAGE_DIR, train_transform)
    val_dataset = PokemonDataset(val_df, IMAGE_DIR, val_transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == 'cuda' else False
    )

    print(f"Training Sample Count: {len(train_df)}")
    print(f"Validation Sample Count: {len(val_df)}")
    
    model = PokemonCNN(num_types=len(ALL_TYPES), num_stats=6).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # optional weight alterations to reduce impact of abundance of trends in pokemon typings
    type_weights = torch.ones(len(ALL_TYPES)).to(DEVICE)
    # type_weights[ALL_TYPES.index("None")] = 0.5
    # type_weights[ALL_TYPES.index("Normal")] = 0.5
    # type_weights[ALL_TYPES.index("Water")] = 0.5
    type_criterion = nn.BCEWithLogitsLoss(pos_weight=type_weights)
    stat_criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    try:
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, 
                                    type_criterion, stat_criterion, DEVICE)
            val_loss = validate(model, val_loader, type_criterion, 
                               stat_criterion, DEVICE)
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f" - Training Loss: {train_loss:.4f}")
            print(f" - Validation Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"New best model saved to {MODEL_SAVE_PATH}")
    
    except KeyboardInterrupt:
        print("\n\nTraining cancelled early by keyboard interrupt")
        print(f"Best validation of {best_val_loss:.4f}")
        if best_val_loss < float('inf'):
            print(f"Best model saved at {MODEL_SAVE_PATH}")
        return
    
    print(f"\nCNN training complete, with the best model saved at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()