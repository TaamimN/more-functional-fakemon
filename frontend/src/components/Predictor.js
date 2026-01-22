import React, { useState } from "react";
import "./Predictor.css";

function Predictor() {
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setImagePreview(URL.createObjectURL(f));
    setResult(null);
  };

  const handlePredict = async () => {
    if (!file) return alert("Please select an image");
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("https://reprints-crystal-reasonably-muslim.trycloudflare.com/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (err) {
      alert("Error calling API: " + err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="predictor-container">
      <div className="upload-section">
        <label className="file-input-label">
          <input type="file" accept="image/png" onChange={handleFileChange} className="file-input" />
          <span className="file-input-button">Choose Image</span>
        </label>
        <button onClick={handlePredict} className="predict-button" disabled={!file || loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </div>

      {imagePreview && (
        <div className="preview-section">
          <h3>Preview</h3>
          <img src={imagePreview} alt="upload preview" className="preview-image" />
        </div>
      )}

      {result && (
        <div className="results-section">
          <div className="types-card">
            <h3>Predicted Types</h3>
            <div className="types-list">
              {result.types.map((t, i) => (
                <div key={i} className="type-item">
                  <span className="type-name">{t.type}</span>
                  <span className="type-confidence">{(t.confidence * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>

          <div className="stats-card">
            <h3>Predicted Stats</h3>
            <div className="stats-grid">
              {Object.entries(result.stats).map(([key, value]) => (
                <div key={key} className="stat-item">
                  <span className="stat-name">{key}</span>
                  <div className="stat-bar-container">
                    <div className="stat-bar" style={{ width: `${(value / 255) * 100}%` }}></div>
                  </div>
                  <span className="stat-value">{value}</span>
                </div>
              ))}
              <div className="stat-total">
                <span>Total</span>
                <span className="total-value">{result.total_stats}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Predictor;