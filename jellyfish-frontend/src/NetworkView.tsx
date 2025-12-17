// src/NetworkView.tsx
import { useEffect, useState } from "react";

type NNDesc = {
  mode?: string;
  has_policy?: boolean;
  input_labels?: string[];
  hidden_size?: number;
  output_labels?: string[];
  layers?: {
    input_to_hidden: {
      weights: number[][]; // [hidden][input]
      bias: number[];
    };
    hidden_to_output: {
      weights: number[][]; // [output][hidden]
      bias: number[];
    };
  };
  has_brain?: boolean;
  message?: string;
};

type NetworkViewProps = {
  inputActs?: number[] | null;
  hiddenActs?: number[] | null;
  hiddenUsage?: number[] | null;
  logits?: number[] | null;
  contrast?: number | null;
};

function neuronColor(a: number | undefined): string {
  if (a === undefined || a === null) {
    return "rgba(148, 163, 184, 0.35)"; // slate-ish
  }
  const v = Math.max(-1, Math.min(1, a));
  if (v >= 0) {
    const alpha = 0.25 + 0.75 * v;
    return `rgba(56, 189, 248, ${alpha})`; // cyan-ish for positive
  } else {
    const alpha = 0.25 + 0.75 * (-v);
    return `rgba(248, 113, 113, ${alpha})`; // red-ish for negative
  }
}

export function NetworkView({
  inputActs,
  hiddenActs,
  hiddenUsage,
  logits,
  contrast,
}: NetworkViewProps) {
  const [nn, setNn] = useState<NNDesc | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch("http://localhost:8000/nn");
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data = (await res.json()) as NNDesc;
        setNn(data);
      } catch (e: any) {
        setError(e?.message ?? "Failed to load /nn");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const inputLabels = nn?.input_labels ?? [];
  const hiddenSize = nn?.hidden_size ?? 0;
  const outputLabels = nn?.output_labels ?? [];

  const width = 380;
  const height = 260;

  // positions
  const xInput = 60;
  const xHidden = 190;
  const xOutput = 320;

  const yForIndex = (idx: number, total: number) => {
    if (total <= 1) return height / 2;
    const padding = 20;
    const span = height - 2 * padding;
    return padding + (span * idx) / (total - 1);
  };

  const hasPolicy = !!nn?.has_policy;

  // Softmax over logits (for output neuron size)
  let probs: number[] | null = null;
  if (logits && logits.length > 0) {
    const maxLogit = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - maxLogit));
    const sum = exps.reduce((a, b) => a + b, 0);
    probs = sum > 0 ? exps.map((e) => e / sum) : null;
  }

  return (
    <div
      style={{
        borderRadius: 8,
        border: "1px solid #1f2937",
        padding: 12,
        background: "#020617",
      }}
    >
      <h3 style={{ margin: "0 0 8px 0", fontSize: 16 }}>Neural Network</h3>

      {loading && (
        <p style={{ fontSize: 12, opacity: 0.7 }}>Loading /nn…</p>
      )}

      {error && (
        <p style={{ fontSize: 12, color: "#fca5a5" }}>
          Error loading /nn: {error}
        </p>
      )}

      {!loading && !error && !hasPolicy && (
        <p style={{ fontSize: 12, opacity: 0.75 }}>
          No policy loaded yet.
          <br />
          <span style={{ opacity: 0.8 }}>
            Hit <code>Reset</code> in the sim after training to load a model.
          </span>
        </p>
      )}

      {hasPolicy && (
        <>
          <svg
            width={width}
            height={height}
            style={{ background: "transparent", display: "block" }}
          >
            {/* edges: input → hidden */}
            {inputLabels.map((_, i) =>
              Array.from({ length: hiddenSize }).map((__, j) => {
                const x1 = xInput;
                const y1 = yForIndex(i, inputLabels.length);
                const x2 = xHidden;
                const y2 = yForIndex(j, hiddenSize);
                return (
                  <line
                    key={`ih-${i}-${j}`}
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="rgba(31, 41, 55, 0.7)"
                    strokeWidth={0.8}
                  />
                );
              })
            )}

            {/* edges: hidden → output */}
            {Array.from({ length: hiddenSize }).map((_, j) =>
              outputLabels.map((__, k) => {
                const x1 = xHidden;
                const y1 = yForIndex(j, hiddenSize);
                const x2 = xOutput;
                const y2 = yForIndex(k, outputLabels.length);
                return (
                  <line
                    key={`ho-${j}-${k}`}
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="rgba(31, 41, 55, 0.7)"
                    strokeWidth={0.8}
                  />
                );
              })
            )}

            {/* input neurons */}
            {inputLabels.map((label, i) => {
              const x = xInput;
              const y = yForIndex(i, inputLabels.length);
              const a = inputActs?.[i];
              const baseR = 7;
              const r = baseR + Math.min(Math.abs(a ?? 0) * 6, 6);
              // map [0,1] into [-1,1] for color so high values glow
              const colorVal =
                a === undefined || a === null ? undefined : a * 2 - 1;

              return (
                <g key={`in-${i}`}>
                  <circle
                    cx={x}
                    cy={y}
                    r={r}
                    fill={neuronColor(colorVal)}
                    style={{ transition: "all 0.15s linear" }}
                  />
                  <text
                    x={x - 10}
                    y={y - 10}
                    fontSize={9}
                    fill="#9ca3af"
                    textAnchor="end"
                  >
                    {label}
                  </text>
                </g>
              );
            })}

            {/* hidden neurons */}
            {Array.from({ length: hiddenSize }).map((_, i) => {
              const x = xHidden;
              const y = yForIndex(i, hiddenSize);
              const a = hiddenActs?.[i];
              const usage = hiddenUsage?.[i] ?? 0;
              const baseR = 7;
              const r = baseR + Math.min(usage * 0.08, 7); // grows with usage, capped
              return (
                <g key={`h-${i}`}>
                  <circle
                    cx={x}
                    cy={y}
                    r={r}
                    fill={neuronColor(a)}
                    stroke="rgba(15, 23, 42, 0.9)"
                    strokeWidth={1}
                    style={{ transition: "all 0.15s linear" }}
                  />
                  <text
                    x={x}
                    y={y + 3}
                    fontSize={8}
                    fill="#e5e7eb"
                    textAnchor="middle"
                    opacity={0.6}
                  >
                    {i}
                  </text>
                </g>
              );
            })}

            {/* output neurons */}
            {outputLabels.map((label, i) => {
              const x = xOutput;
              const y = yForIndex(i, outputLabels.length);
              const prob = probs ? probs[i] : 0;
              const baseR = 9;
              const r = baseR + (prob ?? 0) * 8; // bigger if chosen with high prob
              // map prob [0,1] → [-1,1] for color
              const colorVal =
                prob === undefined || prob === null ? undefined : prob * 2 - 1;

              return (
                <g key={`out-${i}`}>
                  <circle
                    cx={x}
                    cy={y}
                    r={r}
                    fill={neuronColor(colorVal)}
                    stroke="rgba(15, 23, 42, 0.9)"
                    strokeWidth={1}
                    style={{ transition: "all 0.15s linear" }}
                  />
                  <text
                    x={x + 12}
                    y={y + 3}
                    fontSize={9}
                    fill="#e5e7eb"
                    textAnchor="start"
                  >
                    {label}
                  </text>
                </g>
              );
            })}
          </svg>

          <div style={{ fontSize: 11, opacity: 0.8, marginTop: 4 }}>
            <div>
              Inputs & hidden: color = activation (blue: +, red: −), size ∝
              magnitude / usage.
            </div>
            <div>
              Outputs: size ∝ softmax(logit) (action probability).
            </div>
            <div style={{ marginTop: 4 }}>
              current <code>contrast_front</code>:{" "}
              {contrast !== undefined && contrast !== null
                ? contrast.toFixed(3)
                : "-"}
            </div>
            {logits && (
              <div style={{ marginTop: 2 }}>
                logits:{" "}
                {logits.map((v, idx) => (
                  <span key={idx} style={{ marginRight: 4 }}>
                    {nn?.output_labels?.[idx] ?? idx}: {v.toFixed(2)}
                  </span>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
