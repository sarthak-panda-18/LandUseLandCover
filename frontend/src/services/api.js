/**
 * API Service for LULC Backend Integration
 * Handles communication with FastAPI endpoints:
 * - POST /api/classify
 * - GET  /api/model-info
 * - GET  /api/legend
 * - GET  /health
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

/**
 * Checks backend server health and model warmup status.
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!res.ok) {
      throw new Error(`Health check failed with HTTP ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    console.error('API checkHealth error:', error);
    throw new Error(
      error.message || 'Cannot connect to backend server. Make sure the FastAPI service is running on port 8000.'
    );
  }
}

/**
 * Fetches trained Random Forest model metadata, hyperparameters,
 * feature importances, and Phase 5 evaluation test metrics.
 */
export async function getModelInfo() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/model-info`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });

    if (!res.ok) {
      let errDetail = `HTTP ${res.status}`;
      try {
        const errJson = await res.json();
        if (errJson.detail) errDetail = errJson.detail;
      } catch (_) {}
      throw new Error(`Failed to fetch model telemetry (${errDetail})`);
    }

    return await res.json();
  } catch (error) {
    console.error('API getModelInfo error:', error);
    throw error;
  }
}

/**
 * Returns the direct URL to the class color legend PNG image.
 */
export function getLegendUrl() {
  return `${API_BASE_URL}/api/legend`;
}

/**
 * Uploads a Sentinel-2 GeoTIFF (.tif / .tiff) file to be classified by the backend model.
 * 
 * @param {File} file - GeoTIFF raster file
 * @returns {Promise<Object>} Classification result including base64 image, class distribution, and stats.
 */
export async function classifyTile(file) {
  if (!file) {
    throw new Error('Please select a valid GeoTIFF (.tif) file to classify.');
  }

  // Client-side quick validation
  const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
  if (ext !== '.tif' && ext !== '.tiff') {
    throw new Error(`Unsupported file type '${ext}'. Please upload a Sentinel-2 GeoTIFF (.tif, .tiff) file.`);
  }

  if (file.size === 0) {
    throw new Error('The selected file is empty (0 bytes).');
  }

  const MAX_SIZE_MB = 200;
  if (file.size > MAX_SIZE_MB * 1024 * 1024) {
    throw new Error(`File size exceeds the ${MAX_SIZE_MB}MB limit. Please upload a standard Sentinel-2 tile.`);
  }

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_BASE_URL}/api/classify`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let errorMsg = `Server returned HTTP ${res.status}`;
      try {
        const errorData = await res.json();
        if (errorData.detail) {
          errorMsg = errorData.detail;
        }
      } catch (_) {
        const text = await res.text();
        if (text) errorMsg = text;
      }
      throw new Error(errorMsg);
    }

    const data = await res.json();
    return data;
  } catch (error) {
    console.error('API classifyTile error:', error);
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Network error: Unable to reach the backend API at http://127.0.0.1:8000. Is the server running?');
    }
    throw error;
  }
}

/**
 * Returns the list of bundled sample Sentinel-2 tiles.
 */
export async function getSampleTiles() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/sample-tiles`);
    if (!res.ok) return [];
    return await res.json();
  } catch (err) {
    console.warn('Could not fetch sample tiles:', err);
    return [];
  }
}

/**
 * Downloads a sample tile as a File object to classify directly.
 */
export async function fetchSampleTileFile(filename) {
  const res = await fetch(`${API_BASE_URL}/api/sample-tiles/${filename}`);
  if (!res.ok) {
    throw new Error(`Failed to load sample tile: HTTP ${res.status}`);
  }
  const blob = await res.blob();
  return new File([blob], filename, { type: 'image/tiff' });
}

