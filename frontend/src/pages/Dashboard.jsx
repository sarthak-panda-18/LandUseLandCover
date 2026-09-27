import React, { useState, useEffect } from 'react';
import {
  Satellite,
  Activity,
  Layers,
  Sparkles,
  ShieldCheck,
  AlertTriangle,
  RefreshCw,
  Globe,
} from 'lucide-react';

import { checkHealth, getModelInfo, classifyTile } from '../services/api';
import UploadPanel from '../components/UploadPanel';
import ClassifiedMapView from '../components/ClassifiedMapView';
import ClassBreakdownChart from '../components/ClassBreakdownChart';
import ModelStatsPanel from '../components/ModelStatsPanel';

export default function Dashboard() {
  // Backend & Telemetry State
  const [modelInfo, setModelInfo] = useState(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelError, setModelError] = useState(null);
  const [backendHealthy, setBackendHealthy] = useState(false);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);

  // Classification State
  const [classificationResult, setClassificationResult] = useState(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [classifyError, setClassifyError] = useState(null);

  // Fetch telemetry & check backend status on load
  const loadInitialData = async () => {
    setIsCheckingHealth(true);
    setIsModelLoading(true);
    setModelError(null);

    try {
      const healthRes = await checkHealth();
      setBackendHealthy(healthRes?.status === 'ok' && healthRes?.model_loaded);
    } catch (err) {
      setBackendHealthy(false);
      console.warn('Backend health check failed:', err.message);
    } finally {
      setIsCheckingHealth(false);
    }

    try {
      const info = await getModelInfo();
      setModelInfo(info);
    } catch (err) {
      setModelError(err.message || 'Failed to load model telemetry');
    } finally {
      setIsModelLoading(false);
    }
  };

  useEffect(() => {
    loadInitialData();
  }, []);

  // Handler for tile classification request
  const handleClassify = async (file) => {
    setIsClassifying(true);
    setClassifyError(null);

    try {
      const result = await classifyTile(file);
      setClassificationResult(result);
    } catch (err) {
      setClassifyError(err.message || 'An unexpected error occurred during classification.');
    } finally {
      setIsClassifying(false);
    }
  };

  return (
    <div className="dashboard-layout">
      {/* Top Header Navigation */}
      <header className="dashboard-header">
        <div className="header-container">
          <div className="brand-group">
            <div className="logo-badge">
              <Satellite className="logo-icon" size={24} />
            </div>
            <div className="brand-text">
              <h1 className="brand-title">Sentinel-2 LULC AI Engine</h1>
              <p className="brand-subtitle">
                Multispectral Land Use & Land Cover Random Forest Classifier
              </p>
            </div>
          </div>

          <div className="header-status-group">
            <div
              className={`health-pill ${backendHealthy ? 'healthy' : 'unhealthy'}`}
              title={backendHealthy ? 'FastAPI Backend & ML Model Online' : 'Backend offline'}
            >
              <span className="status-dot"></span>
              <span>{backendHealthy ? 'Model Engine Ready' : 'Backend Offline'}</span>
            </div>

            <button
              type="button"
              className="btn-icon-refresh"
              onClick={loadInitialData}
              disabled={isCheckingHealth}
              title="Refresh connection & telemetry"
            >
              <RefreshCw className={isCheckingHealth ? 'spin-icon' : ''} size={16} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="dashboard-main container">
        {/* Offline Alert Banner */}
        {!backendHealthy && !isCheckingHealth && (
          <div className="alert-banner warning">
            <AlertTriangle size={18} className="alert-icon" />
            <div className="alert-content">
              <strong>Backend Connection Notice:</strong> The FastAPI server is not responding at{' '}
              <code>http://127.0.0.1:8000</code>. Please start the backend service (
              <code>uvicorn app.main:app --port 8000</code>) to enable live classification.
            </div>
          </div>
        )}

        {/* Top: Model Performance Telemetry Panel */}
        <section className="dashboard-section section-stats">
          <ModelStatsPanel
            modelInfo={modelInfo}
            isLoading={isModelLoading}
            error={modelError}
          />
        </section>

        {/* Core Interactive Grid */}
        <section className="dashboard-section section-workspace">
          <div className="workspace-grid">
            {/* Left Column: Upload Panel + Class Breakdown */}
            <div className="workspace-left-col">
              <UploadPanel
                onClassify={handleClassify}
                isClassifying={isClassifying}
                error={classifyError}
                onClearError={() => setClassifyError(null)}
              />

              {classificationResult && (
                <ClassBreakdownChart result={classificationResult} />
              )}
            </div>

            {/* Right Column: Classified Map Viewer */}
            <div className="workspace-right-col">
              <ClassifiedMapView result={classificationResult} />
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="dashboard-footer">
        <div className="container footer-content">
          <span>LULC Classification System • Sentinel-2 10m Multispectral Imagery • Random Forest Classifier</span>
        </div>
      </footer>
    </div>
  );
}
