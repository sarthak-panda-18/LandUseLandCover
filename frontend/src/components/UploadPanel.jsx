import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, FileImage, AlertCircle, Play, CheckCircle2, RefreshCw, Layers, Sparkles } from 'lucide-react';
import { getSampleTiles, fetchSampleTileFile } from '../services/api';

export default function UploadPanel({ onClassify, isClassifying, error, onClearError }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [sampleTiles, setSampleTiles] = useState([]);
  const [isLoadingSample, setIsLoadingSample] = useState(false);
  const fileInputRef = useRef(null);

  // Fetch sample tiles list
  useEffect(() => {
    getSampleTiles().then((tiles) => {
      if (tiles && tiles.length > 0) setSampleTiles(tiles);
    });
  }, []);

  // Timer for active inference
  useEffect(() => {
    let interval = null;
    if (isClassifying) {
      setElapsedSeconds(0);
      interval = setInterval(() => {
        setElapsedSeconds((prev) => prev + 1);
      }, 1000);
    } else {
      if (interval) clearInterval(interval);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isClassifying]);

  const handleFileChange = (e) => {
    if (onClearError) onClearError();
    const file = e.target.files?.[0];
    if (file) {
      validateAndSetFile(file);
    }
  };

  const validateAndSetFile = (file) => {
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (ext !== '.tif' && ext !== '.tiff') {
      if (onClearError) onClearError();
      alert(`Invalid file extension "${ext}". Please choose a Sentinel-2 GeoTIFF (.tif or .tiff) file.`);
      return;
    }
    setSelectedFile(file);
  };

  const handleSelectSample = async (tileMeta) => {
    if (onClearError) onClearError();
    setIsLoadingSample(true);
    try {
      const file = await fetchSampleTileFile(tileMeta.filename);
      setSelectedFile(file);
    } catch (err) {
      alert(`Could not load sample tile: ${err.message}`);
    } finally {
      setIsLoadingSample(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (onClearError) onClearError();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      validateAndSetFile(file);
    }
  };

  const handleTriggerUpload = () => {
    if (!selectedFile || isClassifying) return;
    onClassify(selectedFile);
  };

  const handleClear = (e) => {
    e.stopPropagation();
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (onClearError) onClearError();
  };

  // Helper for dynamic loading stage message
  const getLoadingMessage = (seconds) => {
    if (seconds < 3) return 'Uploading & reading 5-band GeoTIFF raster...';
    if (seconds < 10) return 'Standardizing features [B2, B3, B4, B8, NDVI]...';
    if (seconds < 25) return 'Running Random Forest inference across ~1.2M pixels...';
    return 'Finalizing colorized RGBA map & computing class analytics...';
  };

  return (
    <div className="card upload-card">
      <div className="card-header">
        <div className="header-title-group">
          <Layers className="icon-secondary" size={20} />
          <h2>Tile Upload & Inference</h2>
        </div>
        <span className="badge-pill">Sentinel-2 Tile</span>
      </div>

      {error && (
        <div className="error-alert">
          <AlertCircle size={18} className="error-icon" />
          <div className="error-text">
            <strong>Classification Error:</strong> {error}
          </div>
        </div>
      )}

      {/* Drag & Drop Upload Zone */}
      <div
        className={`dropzone ${isDragOver ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''} ${
          isClassifying ? 'processing' : ''
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isClassifying && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".tif,.tiff"
          style={{ display: 'none' }}
          onChange={handleFileChange}
          disabled={isClassifying}
        />

        {isClassifying ? (
          <div className="processing-state">
            <div className="radar-spinner">
              <div className="radar-circle circle-1"></div>
              <div className="radar-circle circle-2"></div>
              <div className="radar-sweep"></div>
              <FileImage className="radar-icon" size={28} />
            </div>
            <div className="processing-title">Classifying Satellite Tile...</div>
            <div className="processing-step">{getLoadingMessage(elapsedSeconds)}</div>
            <div className="processing-timer">
              <RefreshCw className="spin-icon" size={14} />
              <span>Elapsed time: {elapsedSeconds}s</span>
            </div>
          </div>
        ) : selectedFile ? (
          <div className="file-preview-box">
            <div className="file-icon-wrapper">
              <FileImage size={32} className="icon-accent" />
            </div>
            <div className="file-details">
              <div className="file-name" title={selectedFile.name}>
                {selectedFile.name}
              </div>
              <div className="file-meta">
                <span>{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</span>
                <span className="dot-sep">•</span>
                <span>GeoTIFF Multispectral Tile</span>
              </div>
            </div>
            <button
              type="button"
              className="btn-remove-file"
              onClick={handleClear}
              title="Remove file"
            >
              ×
            </button>
          </div>
        ) : (
          <div className="dropzone-prompt">
            <div className="upload-icon-circle">
              <UploadCloud size={32} className="icon-secondary" />
            </div>
            <div className="upload-main-text">
              <strong>Click to upload</strong> or drag and drop
            </div>
            <div className="upload-sub-text">
              Accepts 5-band or 4-band Sentinel-2 GeoTIFF tiles (.tif, .tiff, up to 200MB)
            </div>
          </div>
        )}
      </div>

      {/* Sample Tiles Quick Selector */}
      {sampleTiles.length > 0 && !isClassifying && (
        <div className="sample-tiles-section">
          <div className="sample-tiles-header">
            <Sparkles size={13} className="icon-secondary" />
            <span>Quick Test Sample Tiles:</span>
          </div>
          <div className="sample-tiles-list">
            {sampleTiles.map((t, idx) => (
              <button
                key={t.filename}
                type="button"
                className={`sample-tile-btn ${selectedFile?.name === t.filename ? 'active' : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  handleSelectSample(t);
                }}
                disabled={isLoadingSample}
              >
                Tile {idx + 1} ({t.size_mb} MB)
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Action Bar */}
      <div className="upload-actions">
        <button
          type="button"
          className="btn btn-primary btn-classify"
          onClick={handleTriggerUpload}
          disabled={!selectedFile || isClassifying || isLoadingSample}
        >
          {isClassifying ? (
            <>
              <RefreshCw className="spin-icon" size={18} />
              <span>Processing Tile... ({elapsedSeconds}s)</span>
            </>
          ) : (
            <>
              <Play size={18} fill="currentColor" />
              <span>Run LULC Classification</span>
            </>
          )}
        </button>
      </div>

      {/* Quick helper note */}
      <div className="upload-tip">
        <span className="tip-badge">Tip</span>
        <span>
          Upload any 10m Sentinel-2 tile or pick a sample tile above to classify land cover.
        </span>
      </div>
    </div>
  );
}

