/**
 * API Service for LULC Backend Integration
 * Handles communication with FastAPI endpoints:
 * - POST /api/classify
 * - GET  /api/model-info
 * - GET  /api/legend
 * - GET  /health
 */

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

// Default timeout: 15s (gives Render free-tier instances sufficient time to wake from spin-down)
const DEFAULT_TIMEOUT_MS = 15000;

/**
 * Fetch wrapper with AbortController timeout support.
 * 
 * @param {string} url - Target URL
 * @param {RequestInit} [options={}] - Standard fetch options
 * @param {number} [timeoutMs=15000] - Timeout duration in milliseconds
 * @returns {Promise<Response>}
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return res;
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeoutMs / 1000}s`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Checks backend server health and status.
 * Uses a 15-second timeout and retries once after a 2-second delay upon initial failure.
 * 
 * @param {number} [retries=1] - Number of retry attempts on failure
 * @param {number} [retryDelayMs=2000] - Delay before retrying in milliseconds
 * @returns {Promise<Object>} Backend health payload
 */
export async function checkHealth(retries = 1, retryDelayMs = 2000) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetchWithTimeout(
        `${API_BASE_URL}/health`,
        {
          method: 'GET',
          headers: { 'Accept': 'application/json' },
        },
        DEFAULT_TIMEOUT_MS
      );
      if (!res.ok) {
        throw new Error(`Health check failed with HTTP ${res.status}`);
      }
      return await res.json();
    } catch (error) {
      if (attempt < retries) {
        console.warn(`API checkHealth attempt ${attempt + 1} failed (${error.message}). Retrying in ${retryDelayMs / 1000}s...`);
        await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
      } else {
        console.error('API checkHealth error after retries:', error);
        throw new Error(
          error.message || `Cannot connect to backend server. Make sure the FastAPI service is running at ${API_BASE_URL}.`
        );
      }
    }
  }
}

/**
 * Fetches trained model metadata, hyperparameters,
 * feature importances / feature specs, and evaluation test metrics.
 * 
 * @param {string} modelChoice - 'pixel_rf' | 'rf_patch' | 'efficientnet_patch'
 */
export async function getModelInfo(modelChoice = 'pixel_rf') {
  try {
    const url = new URL(`${API_BASE_URL}/api/model-info`);
    if (modelChoice) {
      url.searchParams.set('model', modelChoice);
    }
    const res = await fetchWithTimeout(
      url.toString(),
      {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      },
      DEFAULT_TIMEOUT_MS
    );

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
    console.error(`API getModelInfo (${modelChoice}) error:`, error);
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
 * Fetches the current status of an asynchronous classification job.
 * 
 * @param {string} jobId - UUID of the classification job
 * @returns {Promise<Object>} Job status payload { job_id, status, result, error, created_at, updated_at }
 */
export async function getJobStatus(jobId) {
  const res = await fetchWithTimeout(
    `${API_BASE_URL}/api/classify/status/${jobId}`,
    {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    },
    DEFAULT_TIMEOUT_MS
  );

  if (res.status === 404) {
    throw new Error('Classification job expired or not found. Please try uploading again.');
  }

  if (!res.ok) {
    let errorMsg = `Server returned HTTP ${res.status}`;
    try {
      const errorData = await res.json();
      if (errorData.detail) errorMsg = errorData.detail;
    } catch (_) {}
    throw new Error(`Failed to check job status: ${errorMsg}`);
  }

  return await res.json();
}

/**
 * Uploads a Sentinel-2 GeoTIFF (.tif / .tiff) file to be classified by the backend model.
 * Uses an asynchronous job pattern: initiates job (202 Accepted) and polls status until complete.
 * 
 * @param {File} file - GeoTIFF raster file
 * @param {string} modelChoice - Selected model ('pixel_rf' | 'rf_patch' | 'efficientnet_patch')
 * @returns {Promise<Object>} Classification result including base64 image, class distribution, and stats.
 */
export async function classifyTile(file, modelChoice = 'pixel_rf') {
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
  formData.append('model', modelChoice || 'pixel_rf');

  try {
    const res = await fetch(`${API_BASE_URL}/api/classify`, {
      method: 'POST',
      body: formData,
    });

    if (res.status === 429) {
      throw new Error('Server is busy processing another request, please try again in a moment.');
    }

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

    const initData = await res.json();
    const jobId = initData?.job_id;
    if (!jobId) {
      throw new Error('Failed to start classification job: No job ID returned from server.');
    }

    // Poll GET /api/classify/status/{jobId} every 2 seconds
    const POLL_INTERVAL_MS = 2000;
    const MAX_POLL_TIME_MS = 10 * 60 * 1000; // 10 minutes timeout limit
    const startTime = Date.now();

    while (true) {
      if (Date.now() - startTime > MAX_POLL_TIME_MS) {
        throw new Error('Classification timed out after 10 minutes. Please try again with a smaller tile or faster model.');
      }

      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));

      let jobData;
      try {
        jobData = await getJobStatus(jobId);
      } catch (pollErr) {
        if (pollErr.message.includes('expired or not found')) {
          throw pollErr;
        }
        console.warn(`Polling status check encountered transient error: ${pollErr.message}. Retrying...`);
        continue;
      }

      const jobStatus = jobData?.status;

      if (jobStatus === 'done') {
        if (!jobData.result) {
          throw new Error('Classification completed but no result payload was returned.');
        }
        return jobData.result;
      }

      if (jobStatus === 'error') {
        throw new Error(jobData.error || 'Classification failed due to an internal server error.');
      }

      if (jobStatus === 'pending' || jobStatus === 'processing') {
        // Still running, continue polling
        continue;
      }

      throw new Error(`Unexpected job status '${jobStatus}' received from server.`);
    }
  } catch (error) {
    console.error('API classifyTile error:', error);
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error(`Network error: Unable to reach the backend API at ${API_BASE_URL}. Is the server running?`);
    }
    throw error;
  }
}

/**
 * Returns the list of bundled sample Sentinel-2 tiles.
 */
export async function getSampleTiles() {
  try {
    const res = await fetchWithTimeout(
      `${API_BASE_URL}/api/sample-tiles`,
      {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      },
      DEFAULT_TIMEOUT_MS
    );
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
  const res = await fetchWithTimeout(
    `${API_BASE_URL}/api/sample-tiles/${filename}`,
    {
      method: 'GET',
    },
    DEFAULT_TIMEOUT_MS
  );
  if (!res.ok) {
    throw new Error(`Failed to load sample tile: HTTP ${res.status}`);
  }
  const blob = await res.blob();
  return new File([blob], filename, { type: 'image/tiff' });
}

