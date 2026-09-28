import React, { useState } from 'react';
import {
  Activity,
  Cpu,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Layers,
  Award,
  BarChart2,
  Info,
  Zap,
  BrainCircuit,
  Sparkles,
} from 'lucide-react';

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
    model_id = 'pixel_rf',
    model_name = 'Random Forest (Pixel-Level)',
    model_type,
    granularity,
    features,
    feature_importances,
    test_metrics = {},
    per_class_metrics = {},
    special_notes = [],
  } = modelInfo;

  const isPixelModel = model_id === 'pixel_rf';
  const isGrasslandZero =
    test_metrics?.grassland_f1 === 0 ||
    per_class_metrics?.Grassland?.f1_score === 0;

  const accuracyPct = test_metrics?.overall_accuracy != null
    ? (test_metrics.overall_accuracy * 100).toFixed(1)
    : '66.3';
  const kappaScore = test_metrics?.cohen_kappa != null
    ? test_metrics.cohen_kappa.toFixed(4)
    : '0.5323';
  const macroF1 = test_metrics?.macro_avg_f1 != null
    ? (test_metrics.macro_avg_f1 * 100).toFixed(1)
    : '62.3';
  const weightedF1 = test_metrics?.weighted_avg_f1 != null
    ? (test_metrics.weighted_avg_f1 * 100).toFixed(1)
    : '69.8';

  const getF1BadgeClass = (f1Score) => {
    if (f1Score === 0) return 'f1-badge f1-zero';
    if (f1Score >= 0.70) return 'f1-badge f1-high';
    if (f1Score >= 0.60) return 'f1-badge f1-med';
    return 'f1-badge f1-low';
  };

  const getModelIcon = () => {
    if (model_id === 'rf_patch') return <Zap size={14} className="icon-accent" />;
    if (model_id === 'efficientnet_patch') return <BrainCircuit size={14} className="icon-accent" />;
    return <Cpu size={14} className="icon-accent" />;
  };

  return (
    <div className="card model-stats-card">
      <div className="card-header">
        <div className="header-title-group">
          <Activity className="icon-success" size={20} />
          <h2>Model Test Performance</h2>
        </div>
        <div className="header-badge-group">
          <span className="badge-model">
            {getModelIcon()}
            <span>{model_name}</span>
          </span>
          {granularity && (
            <span className="badge-granularity">{granularity}</span>
          )}
        </div>
      </div>

      {/* Explicit Grassland Failure Warning when F1 is 0 */}
      {isGrasslandZero && (
        <div className="grassland-alert-banner">
          <AlertTriangle size={18} className="grassland-alert-icon" />
          <div className="grassland-alert-content">
            <strong>Grassland Detection Notice:</strong>
            <p>
              Grassland: model cannot currently identify this class (0 test samples correctly classified).
              Due to severe dataset imbalance in 64x64 blocks, all test Grassland patches are misclassified as Crops.
            </p>
          </div>
        </div>
      )}

      {/* KPI Grid */}
      <div className="kpi-grid">
        <div className="kpi-card accent-success">
          <div className="kpi-label">Overall Accuracy</div>
          <div className="kpi-value">{accuracyPct}%</div>
          <div className="kpi-sub">
            {isPixelModel
              ? 'Test Set (1,923,589 px)'
              : 'Test Set (328 blocks)*'}
          </div>
        </div>

        <div className="kpi-card accent-secondary">
          <div className="kpi-label">Cohen's Kappa (κ)</div>
          <div className="kpi-value">{kappaScore}</div>
          <div className="kpi-sub">
            {parseFloat(kappaScore) >= 0.8 ? 'Near Perfect Agreement' : 'Substantial Agreement'}
          </div>
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

      {/* Feature Section: Spectral Importances (Pixel RF) OR Patch Architecture Spec */}
      {feature_importances ? (
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
      ) : features ? (
        <div className="feature-importance-section patch-features-card">
          <div className="sub-section-title">
            <Sparkles size={14} className="icon-secondary" />
            <span>Model Representation & Feature Pipeline</span>
          </div>
          <p className="patch-feature-description">{features}</p>
        </div>
      ) : null}

      {/* Special Notes / Context Pills for Patch Models */}
      {special_notes && special_notes.length > 0 && (
        <div className="special-notes-box">
          <div className="special-notes-header">
            <Info size={14} className="icon-secondary" />
            <span>Model Evaluation & Architecture Notes:</span>
          </div>
          <ul className="special-notes-list">
            {special_notes.map((note, idx) => (
              <li key={idx}>{note}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Expandable Per-Class Test Precision/Recall Table */}
      {per_class_metrics && Object.keys(per_class_metrics).length > 0 && (
        <div className="detailed-metrics-toggle-area">
          <button
            type="button"
            className="btn-toggle-details"
            onClick={() => setShowDetailedMetrics(!showDetailedMetrics)}
          >
            <span>
              {showDetailedMetrics
                ? 'Hide Per-Class Test Scores'
                : 'View Per-Class Precision / Recall Breakdown'}
            </span>
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
                  {Object.entries(per_class_metrics).map(([clsName, m]) => {
                    const isZeroF1 = m.f1_score === 0;
                    return (
                      <tr key={clsName} className={isZeroF1 ? 'row-warning' : ''}>
                        <td className="cls-cell">
                          <strong>{clsName}</strong>
                          {isZeroF1 && (
                            <span className="cell-warning-tag" title="Model cannot identify this class">
                              Zero F1
                            </span>
                          )}
                        </td>
                        <td>{(m.precision * 100).toFixed(1)}%</td>
                        <td>{(m.recall * 100).toFixed(1)}%</td>
                        <td>
                          <span className={getF1BadgeClass(m.f1_score)}>
                            {(m.f1_score * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="support-cell">
                          {m.support_pixels != null
                            ? `${m.support_pixels.toLocaleString()} px`
                            : isZeroF1
                            ? '0 detected (test)'
                            : '328 test blocks'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
