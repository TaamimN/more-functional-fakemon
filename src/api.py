from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms
from io import BytesIO
import uvicorn
from contextlib import asynccontextmanager

from config import ALL_TYPES, DEVICE, MODEL_SAVE_PATH
from model import PokemonCNN

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = PokemonCNN(num_types=len(ALL_TYPES), num_stats=6)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")
    yield
    print("Terminating")

app = FastAPI(title="More Functional Fakemon API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://more-functional-fakemon.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "More Functional Fakemon API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height
        
        if aspect_ratio > 1:
            new_width = 128
            new_height = int(128 / aspect_ratio)
        else:
            new_height = 128
            new_width = int(128 * aspect_ratio)
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        background = Image.new('RGBA', (128, 128), (128, 128, 128, 255))
        offset_x = (128 - new_width) // 2
        offset_y = (128 - new_height) // 2
        background.paste(image, (offset_x, offset_y), image)
        
        image = background.convert('RGB')
        
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_types_logits, pred_stats = model(img_tensor)
        
        pred_types_probs = torch.sigmoid(pred_types_logits).squeeze().cpu().numpy()
        
        top2_indices = pred_types_probs.argsort()[-2:][::-1]
        predicted_types = []
        for idx in top2_indices:
            if ALL_TYPES[idx] != "None":
                predicted_types.append({
                    "type": ALL_TYPES[idx],
                    "confidence": float(pred_types_probs[idx])
                })
        
        pred_stats = (pred_stats.squeeze().cpu().numpy() * 255).astype(int)
        stat_names = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
        predicted_stats = {
            stat_names[i]: int(pred_stats[i])
            for i in range(6)
        }
        
        return JSONResponse({
            "types": predicted_types,
            "stats": predicted_stats,
            "total_stats": int(pred_stats.sum())
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)