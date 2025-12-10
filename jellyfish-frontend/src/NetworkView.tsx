import { useEffect, useState } from "react";
import type { NetworkDescription } from "./types";

export function NetworkView() {
  const [data, setData] = useState<NetworkDescription | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [polling, setPolling] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const fetchOnce = () => {
      fetch("http://localhost:8000/nn")
        .then((res) => res.json())
        .then((json) => {
          if (cancelled) return;
          setData(json);
          setError(null);

          const hasLayers =
            json &&
            json.has_brain &&
            json.has_policy !== false &&
            json.layers &&
            json.input_labels &&
            json.output_labels;

          if (hasLayers) {
            setPolling(false);
          }
        })
        .catch((err) => {
          if (cancelled) return;
          setError(String(err));
        });
    };

    fetchOnce();
    const id = setInterval(() => {
      if (polling) fetchOnce();
    }, 2000);

    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [polling]);

  // --- render states ---

  if (error) {
    return (
      <div className="nn-panel" style={{ color: "#e5e7eb" }}>
        <h3>Neural Network</h3>
        <p style={{ color: "#f97373", fontSize: 14 }}>Error: {error}</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="nn-panel" style={{ color: "#e5e7eb" }}>
        <h3>Neural Network</h3>
        <p style={{ fontSize: 14 }}>Loading description from backend…</p>
      </div>
    );
  }

  if (!data.has_brain) {
    return (
      <div className="nn-panel" style={{ color: "#e5e7eb" }}>
        <h3>Neural Network</h3>
        <p style={{ fontSize: 14 }}>
          No brain initialized yet. Hit <strong>Reset</strong> in the sim.
        </p>
      </div>
    );
  }

  if (!data.layers || !data.input_labels || !data.output_labels) {
    return (
      <div className="nn-panel" style={{ color: "#e5e7eb" }}>
        <h3>Neural Network</h3>
        <p style={{ fontSize: 14 }}>
          Brain is active but no policy network reported yet.
          <br />
          Make sure a policy file is loaded on reset.
        </p>
      </div>
    );
  }

  // --- we have a full network to draw ---

  const inputs = data.input_labels;
  const hiddenSize = data.hidden_size ?? data.layers.input_to_hidden.bias.length;
  const outputs = data.output_labels;
  const l1 = data.layers.input_to_hidden;
  const l2 = data.layers.hidden_to_output;

  const allW = [...l1.weights.flat(), ...l2.weights.flat()];
  const maxAbs = allW.reduce((m, w) => Math.max(m, Math.abs(w)), 1e-6);

  // Bigger canvas
  const width = 900;
  const height = 520;
  const marginX = 120;
  const layerX = [marginX, width / 2, width - marginX];

  const nodeRadius = 16; // bigger nodes

  const layerYPositions = (count: number) => {
    const top = 60;
    const bottom = height - 60;
    const step = (bottom - top) / Math.max(count - 1, 1);
    return Array.from({ length: count }, (_, i) => top + i * step);
  };

  const inY = layerYPositions(inputs.length);
  const hidY = layerYPositions(hiddenSize);
  const outY = layerYPositions(outputs.length);

  return (
    <div
      className="nn-panel"
      style={{
        marginTop: "1rem",
        color: "#e5e7eb",
      }}
    >
      <h3 style={{ marginBottom: "0.5rem" }}>Neural Network (current brain)</h3>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{
          width: "100%",
          height: "420px", // force a decent on-screen size
          background: "#020617",
          borderRadius: "12px",
          border: "1px solid #1f2937",
        }}
      >
        {/* Input → Hidden connections */}
        {l1.weights.map((row, hIdx) =>
          row.map((w, iIdx) => {
            const mag = Math.abs(w) / maxAbs;
            const strokeWidth = 1 + 4 * mag;
            const color = w >= 0 ? "#22c55e" : "#ef4444"; // brighter
            return (
              <line
                key={`ih-${hIdx}-${iIdx}`}
                x1={layerX[0] + nodeRadius}
                y1={inY[iIdx]}
                x2={layerX[1] - nodeRadius}
                y2={hidY[hIdx]}
                stroke={color}
                strokeWidth={strokeWidth}
                strokeOpacity={0.7}
              />
            );
          })
        )}

        {/* Hidden → Output connections */}
        {l2.weights.map((row, oIdx) =>
          row.map((w, hIdx) => {
            const mag = Math.abs(w) / maxAbs;
            const strokeWidth = 1 + 4 * mag;
            const color = w >= 0 ? "#22c55e" : "#ef4444";
            return (
              <line
                key={`ho-${oIdx}-${hIdx}`}
                x1={layerX[1] + nodeRadius}
                y1={hidY[hIdx]}
                x2={layerX[2] - nodeRadius}
                y2={outY[oIdx]}
                stroke={color}
                strokeWidth={strokeWidth}
                strokeOpacity={0.7}
              />
            );
          })
        )}

        {/* Input nodes */}
        {inputs.map((label, i) => (
          <g key={`in-${i}`}>
            <circle cx={layerX[0]} cy={inY[i]} r={nodeRadius} fill="#2563eb" />
            <text
              x={layerX[0] - (nodeRadius + 10)}
              y={inY[i]}
              textAnchor="end"
              alignmentBaseline="middle"
              fill="#e5e7eb"
              fontSize="13"
            >
              {label}
            </text>
          </g>
        ))}

        {/* Hidden nodes */}
        {Array.from({ length: hiddenSize }, (_, i) => (
          <g key={`hid-${i}`}>
            <circle cx={layerX[1]} cy={hidY[i]} r={nodeRadius} fill="#a855f7" />
            <text
              x={layerX[1]}
              y={hidY[i] - (nodeRadius + 6)}
              textAnchor="middle"
              alignmentBaseline="baseline"
              fill="#e5e7eb"
              fontSize="12"
            >
              h{i}
            </text>
          </g>
        ))}

        {/* Output nodes */}
        {outputs.map((label, i) => (
          <g key={`out-${i}`}>
            <circle cx={layerX[2]} cy={outY[i]} r={nodeRadius} fill="#f97316" />
            <text
              x={layerX[2] + (nodeRadius + 10)}
              y={outY[i]}
              textAnchor="start"
              alignmentBaseline="middle"
              fill="#e5e7eb"
              fontSize="13"
            >
              {label}
            </text>
          </g>
        ))}
      </svg>

      <p style={{ fontSize: "0.85rem", color: "#9ca3af", marginTop: "0.4rem" }}>
        <span style={{ color: "#22c55e" }}>Green</span> = positive weight,&nbsp;
        <span style={{ color: "#ef4444" }}>red</span> = negative. Line thickness ∝ |weight|.
      </p>
    </div>
  );
}
