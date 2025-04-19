import { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [showModelInfo, setShowModelInfo] = useState(false)
  const [historyImage, setHistoryImage] = useState(null)
  const [uploadType, setUploadType] = useState('image') // 'image' or 'video'
  const fileInputRef = useRef(null)
  const videoInputRef = useRef(null)

  // API URL - change this to match your Flask backend URL
  const API_URL = 'http://localhost:5000/api'

  // Fetch model info when component mounts
  useEffect(() => {
    fetchModelInfo()
    fetchHistoryImage()
  }, [])

  const fetchModelInfo = async () => {
    try {
      const response = await fetch(`${API_URL}/info`)
      if (!response.ok) {
        throw new Error('Failed to fetch model information')
      }
      const data = await response.json()
      setModelInfo(data)
    } catch (err) {
      console.error('Error fetching model info:', err.message)
    }
  }

  const fetchHistoryImage = async () => {
    try {
      const response = await fetch(`${API_URL}/model/history`)
      if (!response.ok) {
        return
      }
      const blob = await response.blob()
      const imageUrl = URL.createObjectURL(blob)
      setHistoryImage(imageUrl)
    } catch (err) {
      console.error('Error fetching model history:', err.message)
    }
  }

  const handleFileChange = (event) => {
    setError(null)
    const selectedFile = event.target.files[0]
    
    if (!selectedFile) {
      return
    }
    
    // Check file type based on current mode
    if (uploadType === 'image' && !selectedFile.type.startsWith('image/')) {
      setError('Please select an image file')
      return
    } else if (uploadType === 'video' && !selectedFile.type.startsWith('video/')) {
      setError('Please select a video file')
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
      setError(`Please select an ${uploadType} first`)
      return
    }
    
    setLoading(true)
    setError(null)
    
    try {
      // Create form data
      const formData = new FormData()
      formData.append(uploadType, file)
      
      // Select the appropriate endpoint based on upload type
      const endpoint = uploadType === 'image' ? 'detect' : 'video/detect'
      
      // Make the API call
      const response = await fetch(`${API_URL}/${endpoint}`, {
        method: 'POST',
        body: formData,
      })
      
      // Handle non-200 responses
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || `An error occurred while analyzing the ${uploadType}`)
      }
      
      // Parse the result
      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || `An error occurred while analyzing the ${uploadType}`)
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
    if (videoInputRef.current) {
      videoInputRef.current.value = ''
    }
  }

  const toggleModelInfo = () => {
    setShowModelInfo(!showModelInfo)
  }

  const toggleUploadType = () => {
    // Reset form when changing upload type
    resetForm()
    setUploadType(uploadType === 'image' ? 'video' : 'image')
  }

  return (
    <div className="deepfake-detector">
      <header>
        <h1>DeepFake Detector</h1>
        <p className="subtitle">
          Upload an {uploadType} to detect if it's real or fake using advanced deep learning
        </p>
        <div className="model-info-toggle">
          <button onClick={toggleModelInfo} className="info-button">
            {showModelInfo ? 'Hide' : 'Show'} Model Info
          </button>
        </div>
      </header>

      {showModelInfo && modelInfo && (
        <section className="model-info-section">
          <h2>Model Information</h2>
          <div className="model-info-grid">
            <div className="model-info-item">
              <span className="info-label">Model:</span> 
              <span className="info-value">{modelInfo.model_name}</span>
            </div>
            <div className="model-info-item">
              <span className="info-label">Input Shape:</span>
              <span className="info-value">{modelInfo.input_shape}</span>
            </div>
            <div className="model-info-item">
              <span className="info-label">Parameters:</span>
              <span className="info-value">{modelInfo.total_parameters.toLocaleString()}</span>
            </div>
            <div className="model-info-item">
              <span className="info-label">Detection Threshold:</span>
              <span className="info-value">{modelInfo.threshold.toFixed(4)}</span>
            </div>
          </div>
          
          {historyImage && (
            <div className="model-history">
              <h3>Training History</h3>
              <img src={historyImage} alt="Model Training History" className="history-image" />
            </div>
          )}
        </section>
      )}

      <main>
        <section className="upload-section">
          <div className="upload-type-toggle">
            <button 
              onClick={toggleUploadType} 
              className={uploadType === 'image' ? 'active' : ''}
            >
              Image
            </button>
            <button 
              onClick={toggleUploadType} 
              className={uploadType === 'video' ? 'active' : ''}
            >
              Video (Beta)
            </button>
          </div>
          
          <form onSubmit={handleSubmit}>
            <div className="file-input-container">
              {uploadType === 'image' ? (
                <input 
                  type="file" 
                  accept="image/*" 
                  onChange={handleFileChange}
                  ref={fileInputRef}
                  className="file-input"
                />
              ) : (
                <input 
                  type="file" 
                  accept="video/*" 
                  onChange={handleFileChange}
                  ref={videoInputRef}
                  className="file-input"
                />
              )}
              <button type="button" className="browse-button" onClick={() => 
                uploadType === 'image' ? fileInputRef.current?.click() : videoInputRef.current?.click()
              }>
                Choose {uploadType.charAt(0).toUpperCase() + uploadType.slice(1)}
              </button>
              {file && <span className="file-name">{file.name}</span>}
            </div>

            {previewUrl && (
              <div className="preview-container">
                {uploadType === 'image' ? (
                  <img src={previewUrl} alt="Preview" className="image-preview" />
                ) : (
                  <video 
                    src={previewUrl} 
                    controls 
                    className="video-preview"
                  />
                )}
              </div>
            )}

            <div className="button-container">
              <button 
                type="submit" 
                className="analyze-button"
                disabled={!file || loading}
              >
                {loading ? 'Analyzing...' : `Detect ${uploadType === 'image' ? 'DeepFake' : 'Video Manipulation'}`}
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
                <div 
                  className="threshold-marker" 
                  style={{ left: `${result.threshold_used * 100}%` }}
                  title={`Threshold: ${result.threshold_used.toFixed(4)}`}
                ></div>
              </div>
              <div className="prediction-labels">
                <span>Real</span>
                <span>Fake</span>
              </div>
              <div className="prediction-values">
                <div className="prediction-value">
                  Raw prediction: {result.prediction_value.toFixed(4)}
                </div>
                <div className="threshold-value">
                  Decision threshold: {result.threshold_used.toFixed(4)}
                </div>
              </div>
            </div>
          </section>
        )}

        <section className="info-section">
          <h3>How it works</h3>
          <p>
            This tool uses a deep learning model based on EfficientNetB4 architecture to detect manipulated facial images.
            The model analyzes subtle facial features and identifies patterns consistent with artificially generated or manipulated content.
          </p>
          <p>
            For best results:
          </p>
          <ul>
            <li>Use images with a clear, well-lit face</li>
            <li>The face should be the main subject of the image</li>
            <li>Higher resolution images provide better accuracy</li>
            <li>Video analysis requires the face to be visible in multiple frames</li>
          </ul>
          <div className="technology">
            <p><strong>Technologies used:</strong></p>
            <ul>
              <li>EfficientNetB4 backbone architecture</li>
              <li>MTCNN face detection</li>
              <li>Advanced preprocessing techniques</li>
              <li>Fine-tuned on the Celeb-DF dataset</li>
            </ul>
          </div>
        </section>
      </main>

      <footer>
        <p>© {new Date().getFullYear()} DeepFake Detector Project</p>
        <p className="version">Model version: {modelInfo?.model_name || 'Unknown'}</p>
      </footer>
    </div>
  )
}

export default App
