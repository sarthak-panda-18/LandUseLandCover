import React, { useState } from 'react';
import { Activity, Cpu, CheckCircle2, ChevronDown, ChevronUp, Layers, Award, BarChart2 } from 'lucide-react';

export default function ModelStatsPanel({ modelInfo, isLoading, error }) {
  const [showDetailedMetrics, setShowDetailedMetrics] = useState(false);

  if (isLoading) {
    return (
      <div className="card model-stats-card skeleton-card">
        <div className="skeleton-line" style={{ width: '40%' }}></div>
        <div className="skeleton-grid">
          <div className="skeleton-box"></div>
          <div className="skeleton-box"></div>
          <div className="skeleton-box"></div>
          <div className="skeleton-box"></div>
        </div>
      </div>
    );
  }

  if (error || !modelInfo) {
    return (
      <div className="card model-stats-card">
        <div className="card-header">
          <div className="header-title-group">
            <Activity className="icon-warning" size={20} />
            <h2>Model Test Performance</h2>
          </div>
          <span className="badge-offline">Telemetry Unavailable</span>
        </div>
        <p className="stats-fallback-msg">
          Unable to load offline model metrics ({error || 'Check backend connection'}).
        </p>
      </div>
    );
  }

  const {
    model_type,
    training_samples,
    validation_samples,
    hyperparameters,
    feature_importances,
    test_metrics,
    per_class_metrics,
  } = modelInfo;

  const accuracyPct = test_metrics?.overall_accuracy
    ? (test_metrics.overall_accuracy * 100).toFixed(1)
    : '65.6';
  const kappaScore = test_metrics?.cohen_kappa ? test_metrics.cohen_kappa.toFixed(4) : '0.5267';
  const macroF1 = test_metrics?.macro_avg_f1 ? (test_metrics.macro_avg_f1 * 100).toFixed(1) : '62.6';
  const weightedF1 = test_metrics?.weighted_avg_f1 ? (test_metrics.weighted_avg_f1 * 100).toFixed(1) : '69.6';

  const getF1BadgeClass = (f1Score) => {
    if (f1Score >= 0.70) return 'f1-badge f1-high';
    if (f1Score >= 0.60) return 'f1-badge f1-med';
    return 'f1-badge f1-low';
  };

  return (
    <div className="card model-stats-card">
      <div className="card-header">
        <div className="header-title-group">
          <Activity className="icon-success" size={20} />
          <h2>Model Test Performance</h2>
          <span className="info-tag">Phase 5 Offline Evaluation (1.92M Test Pixels)</span>
        </div>
        <div className="header-badge-group">
          <span className="badge-model">
            <Cpu size={13} />
            <span>Random Forest (300 Trees)</span>
          </span>
        </div>
      </div>

      {/* KPI Grid */}
      <div className="kpi-grid">
        <div className="kpi-card accent-success">
          <div className="kpi-label">Overall Accuracy</div>
          <div className="kpi-value">{accuracyPct}%</div>
          <div className="kpi-sub">Test Set (1,923,589 px)</div>
        </div>

        <div className="kpi-card accent-secondary">
          <div className="kpi-label">Cohen's Kappa (κ)</div>
          <div className="kpi-value">{kappaScore}</div>
          <div className="kpi-sub">Substantial Agreement</div>
        </div>

        <div className="kpi-card accent-warning">
          <div className="kpi-label">Macro Avg F1</div>
          <div className="kpi-value">{macroF1}%</div>
          <div className="kpi-sub">Unweighted Class Mean</div>
        </div>

        <div className="kpi-card accent-success">
          <div className="kpi-label">Weighted Avg F1</div>
          <div className="kpi-value">{weightedF1}%</div>
          <div className="kpi-sub">Support Weighted Mean</div>
        </div>
      </div>

      {/* Feature Importances Mini-Section */}
      {feature_importances && (
        <div className="feature-importance-section">
          <div className="sub-section-title">
            <BarChart2 size={14} className="icon-secondary" />
            <span>Spectral Band Feature Importances</span>
          </div>
          <div className="feature-bars-grid">
            {Object.entries(feature_importances).map(([bandName, imp]) => {
              const impPct = (imp * 100).toFixed(1);
              return (
                <div key={bandName} className="feature-mini-bar">
                  <div className="feature-labels">
                    <span className="band-name">{bandName}</span>
                    <span className="band-pct">{impPct}%</span>
                  </div>
                  <div className="mini-track">
                    <div className="mini-fill" style={{ width: `${imp * 100 * 3.5}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Expandable Per-Class Test Precision/Recall Table */}
      {per_class_metrics && (
        <div className="detailed-metrics-toggle-area">
          <button
            type="button"
            className="btn-toggle-details"
            onClick={() => setShowDetailedMetrics(!showDetailedMetrics)}
          >
            <span>{showDetailedMetrics ? 'Hide Per-Class Test Scores' : 'View Per-Class Precision / Recall Breakdown'}</span>
            {showDetailedMetrics ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
          </button>

          {showDetailedMetrics && (
            <div className="per-class-table-wrapper">
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Test Support</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(per_class_metrics).map(([clsName, m]) => (
                    <tr key={clsName}>
                      <td className="cls-cell">
                        <strong>{clsName}</strong>
                      </td>
                      <td>{(m.precision * 100).toFixed(1)}%</td>
                      <td>{(m.recall * 100).toFixed(1)}%</td>
                      <td>
                        <span className={getF1BadgeClass(m.f1_score)}>
                          {(m.f1_score * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="support-cell">{m.support_pixels?.toLocaleString()} px</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
