import React, { useState } from "react";

function Predictor() {
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setImagePreview(URL.createObjectURL(f));
    setResult(null); // reset previous result
  };

  const handlePredict = async () => {
    if (!file) return alert("Please select an image");
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("https://loan-perhaps-labour-passport.trycloudflare.com/predict", {
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
    <div style={{ marginTop: "1rem" }}>
      <input type="file" accept="image/png" onChange={handleFileChange} />
      <button onClick={handlePredict} style={{ marginLeft: "1rem" }}>
        Predict
      </button>

      {imagePreview && (
        <div style={{ marginTop: "1rem" }}>
          <h3>Preview:</h3>
          <img src={imagePreview} alt="upload preview" width="128" />
        </div>
      )}

      {loading && <p>Loading...</p>}

      {result && (
        <div style={{ marginTop: "1rem" }}>
          <h3>Predicted Types:</h3>
          <ul>
            {result.types.map((t, i) => (
              <li key={i}>
                {t.type} (Confidence: {t.confidence})
              </li>
            ))}
          </ul>

          <h3>Predicted Stats:</h3>
          <table border="1" cellPadding="5">
            <tbody>
              {Object.entries(result.stats).map(([key, value]) => (
                <tr key={key}>
                  <td>{key}</td>
                  <td>{value}</td>
                </tr>
              ))}
              <tr>
                <td><b>Total</b></td>
                <td>{result.total_stats}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default Predictor;
