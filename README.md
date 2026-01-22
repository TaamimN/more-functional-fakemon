# Pokemon Type & Stats Predictor

CNN-based model that predicts Pokemon types and base stats from images. Built for learning AWS deployment, Docker, and APIs

A containerized machine learning API that predicts Pokémon types and base stats from images.
The model is implemented in PyTorch, served via FastAPI, containerized with Docker, and deployed on AWS EC2.

**Frontend:** https://more-functional-fakemon.vercel.app
**API Docs**: http://3.148.255.151/docs 

## Demo

prob a table of some sample images and then json output but id like to make a nicer ui for these ss to exit on first

## Features
- Upload an image of a doodle to predict both its and all 6 stats based on training data of official Pokemon images
- REST API built with FastAPI
- Dockerized for easy deployment
- Hosted on AWS EC2

## Architecture Overvie
- Model: CNN trained in PyTorch to predict Pokémon types and base stats from images
- API: FastAPI service exposing a `/predict` endpoint
- Containerization: Docker image containing model weights and API server
- Backend Deployment: AWS EC2 instance running the Docker container
- Frontend: React application deployed on Vercel that consumes the API
- CORS Handling: Configured to allow requests from the frontend domain

## Local Installation
Frontend is optional and a lightweight React app deployable on Vercel after editing frontend/src/components/Predictor.js fetch to endpoint

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

## Limitations
The dataset used for training was incredibly small (~720 pokemon from generations 1-6)
Pokemon types and stats are hard to predict, as it can be said that Pokemon designs are often intentionally unique therefore broad prediction is not very effective
Most pokemon are monotype, with 17 of the top 20 most abundant type combinations consisting of monotypes
 - Therefore simple CNNs such as this will often blanket prefer to predict monotypes
This simple CNN mostly predicts based on color and vague shape/size
 - Neither of which are especially good at predicting stats, which can be notoriously unpredictable based on design alone

## Future Improvements
This dataset could be improved with the introduction of new/modern pokemon and finer tuning to avoid type combination / stat spread trends
Additionally, it could use the many existing "Fakemon" already created by numerous sources
Methods for identifying more complicated shapes and themes in designs would greatly improve accuracy
Taking in more information (such as a name for the design) along with lexographical analysis would also improve accuracy

## Contributing
Project is currently only for demo purposes. Pull requests are welcome for retraining or improvements

## License
MIT License, refer to LICENSE