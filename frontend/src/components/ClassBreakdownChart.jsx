import React from 'react';
import { BarChart3, PieChart, ShieldCheck, Sparkles } from 'lucide-react';

export default function ClassBreakdownChart({ result }) {
  if (!result || !result.class_distribution) {
    return null;
  }

  const {
    class_distribution,
    valid_pixels,
    total_pixels,
    valid_percentage,
    masked_percentage,
  } = result;

  // Find dominant class
  const dominantClass = [...class_distribution].sort(
    (a, b) => b.pixel_count - a.pixel_count
  )[0];

  // Calculate approximate km² (10m Sentinel-2 pixel = 100 m² = 0.0001 km²)
  const totalAreaKm2 = ((valid_pixels * 100) / 1_000_000).toFixed(2);

  return (
    <div className="card breakdown-card">
      <div className="card-header">
        <div className="header-title-group">
          <BarChart3 className="icon-secondary" size={20} />
          <h2>Class Distribution & Area Coverage</h2>
        </div>
        {dominantClass && (
          <span
            className="badge-dominant"
            style={{ borderColor: dominantClass.color_hex, color: dominantClass.color_hex }}
          >
            <Sparkles size={12} />
            <span>Dominant: {dominantClass.class_name} ({dominantClass.percentage_valid}%)</span>
          </span>
        )}
      </div>

      {/* Composite Multi-Segment Horizontal Stack Bar */}
      <div className="composite-bar-container">
        <div className="composite-bar-label">Tile Land Cover Composition</div>
        <div className="composite-bar">
          {class_distribution.map((item) => {
            if (item.percentage_valid <= 0) return null;
            return (
              <div
                key={item.class_id}
                className="composite-segment"
                style={{
                  width: `${item.percentage_valid}%`,
                  backgroundColor: item.color_hex,
                }}
                title={`${item.class_name}: ${item.percentage_valid}% (${item.pixel_count.toLocaleString()} px)`}
              />
            );
          })}
        </div>
      </div>

      {/* Individual Class Rows */}
      <div className="class-bars-list">
        {class_distribution.map((cls) => {
          const areaKm2 = ((cls.pixel_count * 100) / 1_000_000).toFixed(2);
          return (
            <div key={cls.class_id} className="class-row">
              <div className="class-info-col">
                <div className="class-badge-group">
                  <span
                    className="class-indicator"
                    style={{ backgroundColor: cls.color_hex }}
                  />
                  <span className="class-title">{cls.class_name}</span>
                </div>
                <div className="class-pixel-meta">
                  <span className="class-px-count">{cls.pixel_count.toLocaleString()} px</span>
                  <span className="class-area-km">({areaKm2} km²)</span>
                </div>
              </div>

              <div className="class-bar-col">
                <div className="progress-track">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${cls.percentage_valid}%`,
                      backgroundColor: cls.color_hex,
                    }}
                  />
                </div>
                <span className="class-pct-val">{cls.percentage_valid.toFixed(2)}%</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Summary Metrics Banner */}
      <div className="breakdown-summary-grid">
        <div className="summary-pill">
          <span className="summary-label">Analyzed Area</span>
          <span className="summary-val">{totalAreaKm2} km²</span>
        </div>
        <div className="summary-pill">
          <span className="summary-label">Valid Land Pixels</span>
          <span className="summary-val">{valid_percentage}% ({valid_pixels?.toLocaleString()} px)</span>
        </div>
        <div className="summary-pill">
          <span className="summary-label">Masked/NoData</span>
          <span className="summary-val">{masked_percentage}%</span>
        </div>
      </div>
    </div>
  );
}
