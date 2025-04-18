import { useState, useRef } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  // API URL - change this to match your Flask backend URL
  const API_URL = 'http://localhost:5000/api'

  const handleFileChange = (event) => {
    setError(null)
    const selectedFile = event.target.files[0]
    
    if (!selectedFile) {
      return
    }
    
    // Check if the file is an image
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please select an image file')
      return
    }
    
    setFile(selectedFile)
    
    // Create a preview URL
    const objectUrl = URL.createObjectURL(selectedFile)
    setPreviewUrl(objectUrl)
    
    // Reset result when a new file is selected
    setResult(null)
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    
    if (!file) {
      setError('Please select an image first')
      return
    }
    
    setLoading(true)
    setError(null)
    
    try {
      // Create form data
      const formData = new FormData()
      formData.append('image', file)
      
      // Make the API call
      const response = await fetch(`${API_URL}/detect`, {
        method: 'POST',
        body: formData,
      })
      
      // Handle non-200 responses
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'An error occurred while analyzing the image')
      }
      
      // Parse the result
      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'An error occurred while analyzing the image')
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="deepfake-detector">
      <header>
        <h1>DeepFake Detector</h1>
        <p className="subtitle">
          Upload an image to detect if it's real or fake using advanced deep learning
        </p>
      </header>

      <main>
        <section className="upload-section">
          <form onSubmit={handleSubmit}>
            <div className="file-input-container">
              <input 
                type="file" 
                accept="image/*" 
                onChange={handleFileChange}
                ref={fileInputRef}
                className="file-input"
              />
              <button type="button" className="browse-button" onClick={() => fileInputRef.current?.click()}>
                Choose Image
              </button>
              {file && <span className="file-name">{file.name}</span>}
            </div>

            {previewUrl && (
              <div className="preview-container">
                <img src={previewUrl} alt="Preview" className="image-preview" />
              </div>
            )}

            <div className="button-container">
              <button 
                type="submit" 
                className="analyze-button"
                disabled={!file || loading}
              >
                {loading ? 'Analyzing...' : 'Detect DeepFake'}
              </button>
              {(file || result) && (
                <button 
                  type="button" 
                  className="reset-button"
                  onClick={resetForm}
                >
                  Reset
                </button>
              )}
            </div>
          </form>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </section>

        {result && (
          <section className={`result-section ${result.is_fake ? 'fake' : 'real'}`}>
            <h2 className="result-heading">
              Detection Result
            </h2>
            <div className="result-badge">
              {result.is_fake ? 'FAKE' : 'REAL'}
            </div>
            <div className="confidence">
              Confidence: {result.confidence}%
            </div>
            <div className="prediction-details">
              <div className="prediction-bar-container">
                <div 
                  className="prediction-bar" 
                  style={{ width: `${result.prediction_value * 100}%` }}
                ></div>
              </div>
              <div className="prediction-labels">
                <span>Real</span>
                <span>Fake</span>
              </div>
              <div className="prediction-value">
                Raw prediction: {result.prediction_value.toFixed(4)}
              </div>
            </div>
          </section>
        )}

        <section className="info-section">
          <h3>How it works</h3>
          <p>
            This tool uses a deep learning model based on EfficientNetB0 architecture to detect manipulated facial images.
            The model analyzes facial features and identifies patterns consistent with artificially generated or manipulated content.
          </p>
          <p>
            For best results:
          </p>
          <ul>
            <li>Use images with a clear, well-lit face</li>
            <li>The face should be the main subject of the image</li>
            <li>Higher resolution images provide better accuracy</li>
          </ul>
        </section>
      </main>

      <footer>
        <p>© {new Date().getFullYear()} DeepFake Detector Project</p>
      </footer>
    </div>
  )
}

export default App
