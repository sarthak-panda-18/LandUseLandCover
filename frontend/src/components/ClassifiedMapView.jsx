import React, { useState } from 'react';
import { Map, Download, Eye, Sparkles, Compass, Clock, Grid, Maximize2, Layers } from 'lucide-react';
import { getLegendUrl } from '../services/api';

export default function ClassifiedMapView({ result }) {
  const [viewMode, setViewMode] = useState('classified'); // 'classified', 'rgb', 'split'
  const [showLegendModal, setShowLegendModal] = useState(false);
  const legendUrl = getLegendUrl();

  if (!result) {
    return (
      <div className="card map-empty-card">
        <div className="empty-map-placeholder">
          <div className="empty-icon-pulse">
            <Map size={48} className="icon-muted" />
          </div>
          <h3>No Classified Map Yet</h3>
          <p>Upload a Sentinel-2 GeoTIFF tile on the left and click "Run LULC Classification" to generate high-resolution Land Use and Land Cover maps.</p>
        </div>
      </div>
    );
  }

  const {
    filename,
    dimensions,
    total_pixels,
    valid_pixels,
    valid_percentage,
    processing_time_seconds,
    classified_image_base64,
    rgb_preview_base64,
    geo_metadata,
  } = result;

  const handleDownload = () => {
    if (!classified_image_base64) return;
    const link = document.createElement('a');
    link.href = classified_image_base64;
    link.download = `${filename ? filename.replace(/\.[^/.]+$/, '') : 'tile'}_lulc_classified.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="card map-view-card">
      <div className="card-header">
        <div className="header-title-group">
          <Map className="icon-secondary" size={20} />
          <h2>Classified LULC Map View</h2>
        </div>
        
        <div className="header-actions">
          {/* View Switcher */}
          <div className="view-mode-pills">
            <button
              type="button"
              className={`pill-btn ${viewMode === 'classified' ? 'active' : ''}`}
              onClick={() => setViewMode('classified')}
            >
              <Sparkles size={14} />
              <span>LULC Map</span>
            </button>

            {rgb_preview_base64 && (
              <button
                type="button"
                className={`pill-btn ${viewMode === 'rgb' ? 'active' : ''}`}
                onClick={() => setViewMode('rgb')}
              >
                <Eye size={14} />
                <span>True Color RGB</span>
              </button>
            )}

            {rgb_preview_base64 && (
              <button
                type="button"
                className={`pill-btn ${viewMode === 'split' ? 'active' : ''}`}
                onClick={() => setViewMode('split')}
              >
                <Layers size={14} />
                <span>Side-by-Side</span>
              </button>
            )}
          </div>

          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={handleDownload}
            title="Download classified PNG map"
          >
            <Download size={15} />
            <span>Export PNG</span>
          </button>
        </div>
      </div>

      {/* Main Map Viewer Area */}
      <div className="map-display-container">
        {viewMode === 'classified' && (
          <div className="single-map-view">
            <div className="map-badge">Predicted LULC Classification (6 Classes)</div>
            <img
              src={classified_image_base64}
              alt="Classified LULC Map"
              className="map-raster-image"
            />
          </div>
        )}

        {viewMode === 'rgb' && rgb_preview_base64 && (
          <div className="single-map-view">
            <div className="map-badge">Sentinel-2 True Color Composite (B4, B3, B2)</div>
            <img
              src={rgb_preview_base64}
              alt="Sentinel-2 RGB Preview"
              className="map-raster-image"
            />
          </div>
        )}

        {viewMode === 'split' && rgb_preview_base64 && (
          <div className="split-map-view">
            <div className="split-pane">
              <div className="map-badge">True Color RGB (B4, B3, B2)</div>
              <img
                src={rgb_preview_base64}
                alt="True Color RGB"
                className="map-raster-image"
              />
            </div>
            <div className="split-pane">
              <div className="map-badge">Random Forest LULC Map</div>
              <img
                src={classified_image_base64}
                alt="Classified LULC Map"
                className="map-raster-image"
              />
            </div>
          </div>
        )}
      </div>

      {/* Legend & Meta Footer */}
      <div className="map-footer-grid">
        {/* Class Palette Legend Card */}
        <div className="legend-strip-box">
          <div className="legend-strip-header">
            <span className="legend-title">Class Palette Legend</span>
            <button
              type="button"
              className="btn-link"
              onClick={() => setShowLegendModal(!showLegendModal)}
            >
              {showLegendModal ? 'Hide Graphic' : 'Show Full Legend'}
            </button>
          </div>

          <div className="legend-swatches">
            <div className="legend-item">
              <span className="color-dot" style={{ backgroundColor: '#1E88E5' }}></span>
              <span className="class-name">0: Water</span>
            </div>
            <div className="legend-item">
              <span className="color-dot" style={{ backgroundColor: '#2E7D32' }}></span>
              <span className="class-name">1: Trees/Forest</span>
            </div>
            <div className="legend-item">
              <span className="color-dot" style={{ backgroundColor: '#81C784' }}></span>
              <span className="class-name">2: Crops</span>
            </div>
            <div className="legend-item">
              <span className="color-dot" style={{ backgroundColor: '#DCE775' }}></span>
              <span className="class-name">3: Grassland</span>
            </div>
            <div className="legend-item">
              <span className="color-dot" style={{ backgroundColor: '#E53935' }}></span>
              <span className="class-name">4: Built-up</span>
            </div>
            <div className="legend-item">
              <span className="color-dot" style={{ backgroundColor: '#8D6E63' }}></span>
              <span className="class-name">5: Bare land</span>
            </div>
          </div>

          {showLegendModal && (
            <div className="legend-image-container">
              <img src={legendUrl} alt="LULC Classification Legend Graphic" className="legend-full-img" />
            </div>
          )}
        </div>

        {/* Spatial Metadata Info Box */}
        <div className="spatial-meta-box">
          <div className="meta-item">
            <Grid size={15} className="icon-secondary" />
            <div className="meta-text">
              <span className="meta-label">Dimensions</span>
              <span className="meta-val">
                {dimensions?.width} × {dimensions?.height} px ({total_pixels?.toLocaleString()} total)
              </span>
            </div>
          </div>

          <div className="meta-item">
            <Clock size={15} className="icon-accent" />
            <div className="meta-text">
              <span className="meta-label">Inference Time</span>
              <span className="meta-val">{processing_time_seconds}s</span>
            </div>
          </div>

          <div className="meta-item">
            <Compass size={15} className="icon-primary" />
            <div className="meta-text">
              <span className="meta-label">Spatial CRS</span>
              <span className="meta-val">{geo_metadata?.crs || 'EPSG:32644 (UTM 44N)'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
