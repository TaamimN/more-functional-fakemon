import React from "react";
import Predictor from "./components/Predictor";
import "./App.css";

function App() {
  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Fakemon Type & Stats Predictor</h1>
        <p className="subtitle">Predict Pokemon types and base stats from images using deep learning</p>
      </header>
      <main>
        <Predictor />
      </main>
      <footer className="app-footer">
        <p>Built with PyTorch, FastAPI, and React • Deployed on AWS EC2</p>
      </footer>
    </div>
  );
}

export default App;