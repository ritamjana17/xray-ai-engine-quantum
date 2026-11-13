"use client";

import { useEffect, useState } from "react";

type QuantumResult = {
  prediction: "NORMAL" | "FRACTURED";
  confidence: number;
  probabilities: { NORMAL: number; FRACTURED: number };
  model: string;
  gradcam?: string | null;
  note?: string;
};

export default function QuantumAnalysis({ imageBase64 }: { imageBase64: string }) {
  const [result, setResult] = useState<QuantumResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    async function run() {
      setLoading(true);
      setError(null);
      setResult(null);
      try {
        const response = await fetch("/api/quantum", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ imageBase64 }),
          signal: controller.signal,
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || `Quantum service error (${response.status})`);
        if (active) setResult(data as QuantumResult);
      } catch (err) {
        if (active && !(err instanceof Error && err.name === "AbortError")) {
          setError(err instanceof Error ? err.message : "Quantum analysis failed.");
        }
      } finally {
        if (active) setLoading(false);
      }
    }

    run();
    return () => { active = false; controller.abort(); };
  }, [imageBase64]);

  return (
    <section className="mt-5 rounded-2xl border border-line bg-surface p-5" aria-label="Hybrid quantum-classical analysis">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-accent">Experimental model</p>
          <h2 className="mt-1 font-display text-2xl">Hybrid Quantum Analysis</h2>
          <p className="mt-1 text-sm text-muted">DenseNet121 + 4-qubit PennyLane classifier</p>
        </div>
        <span className="rounded-full border border-line px-3 py-1 text-xs text-muted">Research prototype</span>
      </div>

      {loading && <p className="mt-4 text-sm text-muted">Running the hybrid classifier…</p>}
      {error && (
        <div className="mt-4 rounded-xl border border-line bg-surface-2 p-4 text-sm text-muted">
          <p className="font-semibold text-fg">Quantum service unavailable</p>
          <p className="mt-1">{error}</p>
          <p className="mt-2 text-xs">The existing Gemini analysis remains available. Configure and deploy the Python service to enable this section.</p>
        </div>
      )}
      {result && (
        <>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <div className="rounded-xl bg-surface-2 p-4">
              <p className="text-xs text-muted">Predicted class</p>
              <p className="mt-1 font-display text-2xl">{result.prediction === "FRACTURED" ? "Possible fracture" : "Normal"}</p>
            </div>
            <div className="rounded-xl bg-surface-2 p-4">
              <p className="text-xs text-muted">Model score</p>
              <p className="mt-1 font-display text-2xl">{(result.confidence * 100).toFixed(1)}%</p>
            </div>
          </div>
          <div className="mt-3 space-y-2 text-sm">
            {(["NORMAL", "FRACTURED"] as const).map((label) => (
              <div key={label} className="flex items-center gap-3">
                <span className="w-24 text-muted">{label === "NORMAL" ? "Normal" : "Fractured"}</span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-fill">
                  <div className="h-full rounded-full bg-accent" style={{ width: `${Math.max(0, Math.min(100, result.probabilities[label] * 100))}%` }} />
                </div>
                <span className="w-12 text-right tabular-nums text-muted">{(result.probabilities[label] * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
          {result.gradcam && (
            <div className="mt-5">
              <p className="mb-2 text-sm font-semibold">Grad-CAM attention map</p>
              {/* Service returns a data URL generated from the same input image. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={result.gradcam} alt="Grad-CAM heatmap overlay" className="w-full rounded-xl border border-line" />
            </div>
          )}
          <p className="mt-4 text-xs leading-relaxed text-muted">{result.note || "Experimental model output only. Not a medical diagnosis. A qualified radiologist must review the original image."}</p>
        </>
      )}
    </section>
  );
}
