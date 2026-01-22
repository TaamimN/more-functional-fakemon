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
        <div className="info-section">
          <div className="info-card">
            <h3>API Not Working?</h3>
            <p>If predictions fail, the Cloudflare tunnel may be down. You can still access the API directly:</p>
            <a href="http://3.148.255.151/docs" target="_blank" rel="noopener noreferrer" className="info-link">
              View API Docs
            </a>
          </div>
          <div className="info-card">
            <h3>How It Works</h3>
            <p>Check out the source code and architecture details on GitHub:</p>
            <a href="https://github.com/TaamimN/more-functional-fakemon#" target="_blank" rel="noopener noreferrer" className="info-link">
              View on GitHub
            </a>
          </div>
        </div>
      </main>
      <footer className="app-footer">
        <p>Built with PyTorch, FastAPI, and React • Deployed on AWS EC2</p>
      </footer>
    </div>
  );
}

export default App;