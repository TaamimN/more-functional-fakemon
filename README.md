# Pokemon Type & Stats Predictor

CNN-based model that predicts Pokemon types and base stats from Fakemon, which is a term for any fake or custom created Pokemon. Demonstrates AWS deployment, Docker, and APIs

A containerized machine learning API that predicts Pokémon types and base stats from images.
The model is implemented in PyTorch, served via FastAPI, containerized with Docker, and deployed on AWS EC2.

**Frontend:** https://more-functional-fakemon.vercel.app
**API Docs**: http://3.148.255.151/docs 

## Demo

prob a table of some sample images and then json output but id like to make a nicer ui for these ss to exit on first

## Tech Stack
- **Model**: PyTorch CNN (4-layer architecture)
- **API**: FastAPI with automatic OpenAPI docs
- **Containerization**: Docker
- **Backend**: AWS EC2 (t2.micro)
- **Frontend**: React + Vercel
- **Tunnel**: Cloudflare for HTTPS

## Model Details
- Architecture: Custom CNN with separate heads for type and stat prediction
- Training: ~720 Pokemon images (Generations 1-6)
- Input: 128x128 RGB images
- Output: Multi-label type classification + 6 base stat regression

## Local Installation
Frontend is optional and a lightweight React app deployable on Vercel after editing frontend/src/components/Predictor.js fetch to endpoint.

### Docker Image
```bash
docker pull entroxx/more_functional_fakemon:latest
docker run -d -p 8000:8000 entroxx/more_functional_fakemon:latest
```
Visit `http://localhost:8000/docs` to test endpoints in Swagger UI

### Local Development / Custom Training
Note that weights are not committed but may be found at:
- https://drive.google.com/file/d/1ms-bdnI7Hx-hht0-FQGOUP3Nh88ZSyhF/view?usp=sharing
Place file in root directory of folder
```bash
git clone https://github.com/TaamimN/more-functional-fakemon.git
cd more-functional-fakemon
pip install -r requirements.txt
```
To train, evaluate model, or run the API you may simply run 
- src/train.py (note that no training data is committed)
- src/evaluate.py
- src/api.py
Note that you may want to additionally install pytorch-cuda if using an nvidia gpu when training/evaluating and may regardless wish to edit src/config.py

### `POST /predict`
Upload an image file to get predicted types and stats

**Request:**
- multipart/form-data
  - file: Pokemon image (.png) (ideally transparent background)

**Response:**
```json
{
  "types": [
    {"type": "Water", "confidence": 0.95},
    {"type": "Flying", "confidence": 0.87}
  ],
  "stats": {
    "HP": 80,
    "Attack": 85,
    "Defense": 75,
    "Sp. Atk": 90,
    "Sp. Def": 70,
    "Speed": 110
  },
  "total_stats": 510
}
```

## Deployment Notes
- The backend runs in a Docker container on AWS EC2.
- Model weights are not committed to the repository and must be downloaded separately.
- The frontend is deployed independently on Vercel.
- The frontend supports mock responses for offline or demo-only usage.

## Model Limitations
The model faces several inherent challenges:
**Dataset Size**: Training on only 720 images limits the model's ability to generalize. Modern Pokemon and fan-created designs would improve coverage.
**Type Imbalance**: ~50% of Pokemon are mono-type, causing the model to favor single-type predictions. The top 20 type combinations include 17 mono-types.
**Feature Learning**: The CNN primarily learns color-based patterns with limited shape understanding. This works reasonably for types (blue = Water) but struggles with stats, which don't correlate strongly with visual features.
**Design Philosophy**: Pokemon designs intentionally subvert expectations (some Pokemon base stat distribution is particularly polar, such as Shuckle), making visual-only prediction fundamentally difficult.

## Future Improvements
This dataset could be improved with the introduction of new/modern pokemon and finer tuning to avoid type combination / stat spread trends
Additionally, it could use the many existing "Fakemon" already created by numerous sources
Methods for identifying more complicated shapes and themes in designs would greatly improve accuracy
Taking in more information (such as a name for the design) along with lexographical analysis would also improve accuracy

## Contributing
Project is currently only for demo purposes. Pull requests are welcome for retraining or improvements

## License
MIT License, refer to LICENSE