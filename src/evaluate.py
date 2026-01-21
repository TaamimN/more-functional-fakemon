import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from config import *
from dataset import PokemonDataset
from model import PokemonCNN
from transforms import val_transform
from utils import prepare_dataframe

def evaluate_model(model_path=MODEL_SAVE_PATH):
    print(f"Starting evaluation using {DEVICE}")
    
    df = pd.read_csv(DATA_CSV)
    df = prepare_dataframe(df)

    train_df, val_df = train_test_split(df, test_size=1-TRAIN_SPLIT, random_state=42)
    print(f"Evaluating on {len(val_df)} validation samples")
    
    val_dataset = PokemonDataset(val_df, IMAGE_DIR, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = PokemonCNN(num_types=len(ALL_TYPES), num_stats=6)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    all_type_preds, all_type_targets = [], []
    all_stat_preds, all_stat_targets = [], []
    all_identifiers = []
    with torch.no_grad():
        for batch_idx, (imgs, types, stats) in enumerate(val_loader):
            imgs = imgs.to(DEVICE)

            pred_types, pred_stats = model(imgs)
            pred_types_probs = torch.sigmoid(pred_types).cpu().numpy()
            pred_types_bin = np.zeros_like(pred_types_probs, dtype=int)
            for i in range(len(pred_types_probs)):
                top2_indices = np.argsort(pred_types_probs[i])[::-1][:2]
                pred_types_bin[i, top2_indices] = 1

            all_type_preds.append(pred_types_bin)
            all_type_targets.append(types.int().numpy())
            all_stat_preds.append(pred_stats.cpu().numpy() * 255)
            all_stat_targets.append(stats.numpy() * 255)
            
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min(batch_start + len(imgs), len(val_df))
            all_identifiers.extend(val_df.iloc[batch_start:batch_end]['Identifier'].tolist())
    
    all_type_preds = np.concatenate(all_type_preds, axis=0)
    all_type_targets = np.concatenate(all_type_targets, axis=0)
    all_stat_preds = np.concatenate(all_stat_preds, axis=0)
    all_stat_targets = np.concatenate(all_stat_targets, axis=0)
    
    none_idx = ALL_TYPES.index("None")
    num_with_none = np.sum(all_type_preds[:, none_idx] == 1)
    num_dual = len(all_type_preds) - num_with_none
    
    print(f"\nOverall Type Distribution:")
    print(f" - Monotype:  {num_with_none}/{len(all_type_preds)} ({num_with_none/len(all_type_preds)*100:.1f}%)")
    print(f" - Dual-type: {num_dual}/{len(all_type_preds)} ({num_dual/len(all_type_preds)*100:.1f}%)")
    print()
    
    type_counts = {t: 0 for t in ALL_TYPES if t != "None"}
    for i in range(len(all_type_preds)):
        for j in range(len(ALL_TYPES)):
            if all_type_preds[i][j] == 1 and ALL_TYPES[j] != "None":
                type_counts[ALL_TYPES[j]] += 1
    
    print("Type Prediction Distribution:")
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    for type_name, count in sorted_types:
        percentage = (count / len(all_type_preds)) * 100
        print(f" - {type_name:10s}: {count:3d} ({percentage:5.1f}%)")
    print()
    
    f1 = f1_score(all_type_targets, all_type_preds, average='micro')
    perfect_match = np.mean(np.all(all_type_targets == all_type_preds, axis=1))
    
    mae = np.mean(np.abs(all_stat_preds - all_stat_targets))
    rmse = np.sqrt(np.mean((all_stat_preds - all_stat_targets)**2))
    
    print("Evaluation Results:")
    print(f"\nType Prediction:")
    print(f" - Micro F1 Score: {f1:.4f}")
    print(f" - Perfect Match:    {perfect_match*100:.2f}%")
    print(f"\nStat Prediction:")
    print(f" - MAE:  {mae:.2f}")
    print(f" - RMSE: {rmse:.2f}")
    
    print("Sample Predictions:")
    none_idx = ALL_TYPES.index("None")
    for i in range(min(5, len(all_type_preds))):
        pred_types = [ALL_TYPES[j] for j, v in enumerate(all_type_preds[i]) if v == 1 and j != none_idx]
        true_types = [ALL_TYPES[j] for j, v in enumerate(all_type_targets[i]) if v == 1 and j != none_idx]
        
        print(f"\nPokemon/Image: {all_identifiers[i]}")
        print(f"  Types - Predicted: {pred_types}")
        print(f"          Actual:    {true_types}")
        print(f"  Stats - Predicted: {all_stat_preds[i].astype(int)}")
        print(f"          Actual:    {all_stat_targets[i].astype(int)}")

if __name__ == "__main__":
    evaluate_model()