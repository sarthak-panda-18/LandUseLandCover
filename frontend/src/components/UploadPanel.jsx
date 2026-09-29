import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, FileImage, AlertCircle, Play, CheckCircle2, RefreshCw, Layers, Sparkles, Cpu, Zap, BrainCircuit, Info } from 'lucide-react';
import { getSampleTiles, fetchSampleTileFile } from '../services/api';

const MODEL_OPTIONS = [
  {
    id: 'pixel_rf',
    title: 'Pixel-level (Random Forest)',
    subtitle: 'Full detail, ~53s, 66% accuracy',
    icon: Layers,
    badge: '10m Pixel Resolution',
  },
  {
    id: 'rf_patch',
    title: 'Patch-level (Random Forest)',
    subtitle: 'Fast, ~0.4s, 89% accuracy*',
    icon: Zap,
    badge: '64x64 Block',
  },
  {
    id: 'efficientnet_patch',
    title: 'Patch-level (EfficientNetB0)',
    subtitle: 'Deep learning, ~10s, 87% accuracy*',
    icon: BrainCircuit,
    badge: '64x64 Block',
  },
];

export default function UploadPanel({
  onClassify,
  isClassifying,
  error,
  onClearError,
  selectedModel = 'pixel_rf',
  onSelectModel,
}) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [sampleTiles, setSampleTiles] = useState([]);
  const [loadingSampleFilename, setLoadingSampleFilename] = useState(null);
  const [sampleFetchSeconds, setSampleFetchSeconds] = useState(0);
  const [sampleError, setSampleError] = useState(null);
  const fileInputRef = useRef(null);

  const isLoadingSample = Boolean(loadingSampleFilename);

  // Fetch sample tiles list
  useEffect(() => {
    getSampleTiles().then((tiles) => {
      if (tiles && tiles.length > 0) setSampleTiles(tiles);
    });
  }, []);

  // Timer for sample tile download tracking (shows warning after 5s)
  useEffect(() => {
    let interval = null;
    if (loadingSampleFilename) {
      setSampleFetchSeconds(0);
      interval = setInterval(() => {
        setSampleFetchSeconds((prev) => prev + 1);
      }, 1000);
    } else {
      setSampleFetchSeconds(0);
      if (interval) clearInterval(interval);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [loadingSampleFilename]);

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
    setSampleError(null);
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
    setSampleError(null);
    setLoadingSampleFilename(tileMeta.filename);
    try {
      const file = await fetchSampleTileFile(tileMeta.filename);
      setSelectedFile(file);
    } catch (err) {
      setSampleError(err.message || 'Failed to download sample tile from server.');
    } finally {
      setLoadingSampleFilename(null);
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
    setSampleError(null);
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
    setSampleError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (onClearError) onClearError();
  };

  // Helper for dynamic loading stage message based on active model
  const getLoadingMessage = (seconds, model) => {
    if (model === 'rf_patch') {
      if (seconds < 1) return 'Extracting 64x64 blocks & computing 25 statistical features...';
      if (seconds < 3) return 'Running fast Patch Random Forest inference...';
      return 'Colorizing patch predictions & generating land cover statistics...';
    }
    if (model === 'efficientnet_patch') {
      if (seconds < 3) return 'Extracting 64x64 blocks & preparing RGB composite batches...';
      if (seconds < 8) return 'Running EfficientNetB0 neural network forward pass...';
      return 'Assembling patch tensor predictions into full classification map...';
    }
    // Default pixel_rf
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
            <div className="processing-step">{getLoadingMessage(elapsedSeconds, selectedModel)}</div>
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
            {sampleTiles.map((t, idx) => {
              const isLoadingThis = loadingSampleFilename === t.filename;
              const isSelected = selectedFile?.name === t.filename;

              return (
                <button
                  key={t.filename}
                  type="button"
                  className={`sample-tile-btn ${isSelected ? 'active' : ''} ${isLoadingThis ? 'loading' : ''}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleSelectSample(t);
                  }}
                  disabled={isLoadingSample}
                  title={isLoadingThis ? 'Downloading sample GeoTIFF tile...' : `Select Tile ${idx + 1}`}
                >
                  {isLoadingThis ? (
                    <>
                      <RefreshCw className="spin-icon" size={12} />
                      <span>Loading...</span>
                    </>
                  ) : (
                    `Tile ${idx + 1} (${t.size_mb} MB)`
                  )}
                </button>
              );
            })}
          </div>

          {/* 5-second slow fetch warning */}
          {loadingSampleFilename && sampleFetchSeconds >= 5 && (
            <div className="sample-fetch-notice">
              <RefreshCw className="spin-icon" size={13} />
              <span>
                Downloading tile from server, this can take up to a minute on the free hosting tier.
              </span>
            </div>
          )}

          {/* Inline Sample Tile Error Display */}
          {sampleError && (
            <div className="error-alert sample-error-alert">
              <AlertCircle size={16} className="error-icon" />
              <div className="error-text">
                <strong>Sample Tile Error:</strong> {sampleError}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Model Selection Segmented Group / Radio Cards */}
      <div className="model-selector-section">
        <div className="model-selector-header">
          <span className="selector-title">Select Classification Model</span>
          <span className="selector-count">{MODEL_OPTIONS.length} Models Available</span>
        </div>

        <div className="model-options-list" role="radiogroup" aria-label="Classification Model">
          {MODEL_OPTIONS.map((opt) => {
            const isSelected = selectedModel === opt.id;
            const IconComponent = opt.icon;
            return (
              <div
                key={opt.id}
                role="radio"
                aria-checked={isSelected}
                tabIndex={isClassifying ? -1 : 0}
                className={`model-option-card ${isSelected ? 'selected' : ''} ${
                  isClassifying ? 'disabled' : ''
                }`}
                onClick={() => !isClassifying && onSelectModel && onSelectModel(opt.id)}
                onKeyDown={(e) => {
                  if ((e.key === ' ' || e.key === 'Enter') && !isClassifying && onSelectModel) {
                    e.preventDefault();
                    onSelectModel(opt.id);
                  }
                }}
              >
                <div className="model-radio-indicator">
                  <div className={`radio-outer ${isSelected ? 'active' : ''}`}>
                    {isSelected && <div className="radio-inner" />}
                  </div>
                </div>

                <div className="model-card-content">
                  <div className="model-card-top">
                    <div className="model-title-wrap">
                      <IconComponent size={16} className="model-icon" />
                      <span className="model-title-text">{opt.title}</span>
                    </div>
                    <span className="model-badge">{opt.badge}</span>
                  </div>
                  <div className="model-subtitle-text">{opt.subtitle}</div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Small Asterisk Note */}
        <div className="model-selector-note">
          <Info size={13} className="note-icon" />
          <span>
            *Evaluated on a much smaller test set (328 blocks vs 1.9M pixels) — less statistically robust than the pixel-level accuracy figure.
          </span>
        </div>
      </div>

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

