import os
import torch
from torch.utils.data import Dataset
from PIL import Image

# dataset for df and images+transformations
class PokemonDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['Identifier']}.png")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        types = torch.tensor(row["type_vector"], dtype=torch.float32)
        stats = torch.tensor(row["stats"], dtype=torch.float32) / 255.0
        return img, types, stats