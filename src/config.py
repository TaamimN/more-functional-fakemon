import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR.endswith('src'):
    BASE_DIR = os.path.dirname(BASE_DIR)

DATA_CSV = os.path.join(BASE_DIR, 'data', 'pokemon_data.csv')
IMAGE_DIR = os.path.join(BASE_DIR, 'data', 'training_images')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'pokemon_model_best.pth')

BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
IMG_SIZE = 128

TRAIN_SPLIT = 0.8
STAT_LOSS_WEIGHT = 0.01
NUM_WORKERS = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ALL_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy", "None"
]