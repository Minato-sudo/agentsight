import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const SAMPLE = {
  query: "Move 'final_report.pdf' to the temp directory",
  trajectory: [
    {
      step: 1,
      content: "I need to make a temp directory and move the file there.",
      tool_calls: [{ name: "mkdir", arguments: { dir_name: "temp" } }],
      tool_responses: ["Directory created successfully."]
    },
    {
      step: 2,
      content: "Now I will move the file to the temp directory.",
      tool_calls: [{ name: "mv", arguments: { source: "final_report.pdf", destination: "temp/" } }],
      tool_responses: ["{'error': 'No such file or directory: final_report.pdf'}"]
    },
    {
      step: 3,
      content: "The file has been moved successfully to temp/final_report.pdf.",
      tool_calls: [],
      tool_responses: []
    }
  ]
};

function probColor(p) {
  if (p > 0.6) return '#ef4444';
  if (p > 0.3) return '#f59e0b';
  return '#10b981';
}

function probClass(p) {
  if (p > 0.6) return 'high';
  if (p > 0.3) return 'mid';
  return 'low';
}

export default function App() {
  const [jsonInput, setJsonInput] = useState(JSON.stringify(SAMPLE, null, 2));
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [serverOk, setServerOk] = useState(null);

  // Check server health on mount
  useEffect(() => {
    axios.get(`${API_BASE}/health`, { timeout: 15000 })
      .then(r => { setServerOk(true); setModelInfo(r.data); })
      .catch(() => setServerOk(false));

    axios.get(`${API_BASE}/model/info`, { timeout: 15000 })
      .then(r => setModelInfo(r.data))
      .catch(() => { });
  }, []);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    let payload;
    try {
      payload = JSON.parse(jsonInput);
    } catch {
      setError('Invalid JSON — check your syntax.');
      setLoading(false);
      return;
    }

    try {
      const { data } = await axios.post(`${API_BASE}/analyze`, payload, { timeout: 60000 });
      setResult(data);
    } catch (err) {
      if (err.code === 'ECONNREFUSED' || err.code === 'ERR_NETWORK') {
        setError('Cannot reach the backend. Start it with:\n\nvenv/bin/uvicorn product.api_server:app --reload --port 8000');
      } else {
        setError(err.response?.data?.detail || err.message || 'Unknown error');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo-icon">👁</div>
          <div>
            <h1>AgentSight Monitor</h1>
          </div>
          <div className="header-sub">
            {serverOk === true && <><span className="status-dot" />Backend online</>}
            {serverOk === false && <span style={{ color: '#ef4444' }}>⚠ Backend offline</span>}
            {serverOk === null && <span style={{ color: '#64748b' }}>Connecting…</span>}
          </div>
        </div>
      </header>

      {/* ── Main grid ── */}
      <div className="container">

        {/* ── Input panel ── */}
        <div className="card input-panel">
          <div className="card-title">
            <span>📋</span> Trajectory Input
          </div>

          <textarea
            className="trajectory-editor"
            value={jsonInput}
            onChange={e => setJsonInput(e.target.value)}
            spellCheck={false}
            placeholder='{"query": "...", "trajectory": [...]}'
          />

          <button
            className="analyze-btn"
            onClick={handleAnalyze}
            disabled={loading || serverOk === false}
            id="analyze-btn"
          >
            {loading
              ? <><div className="spinner" /> Analyzing…</>
              : <><span>▶</span> Analyze Trajectory</>
            }
          </button>

          {error && (
            <div className="alert error" style={{ marginTop: '1rem', whiteSpace: 'pre-wrap' }}>
              <span className="alert-icon">⚠</span>
              <span>{error}</span>
            </div>
          )}

          {/* ── Model info card ── */}
          {modelInfo && (
            <div style={{ marginTop: '1.25rem' }}>
              <div className="card-title"><span>⚙</span> Model Info</div>
              <div className="info-grid">
                <div className="info-row">
                  <span className="info-label">Threshold</span>
                  <span className="info-value">{modelInfo.threshold ?? '0.40'}</span>
                </div>
                <div className="info-row">
                  <span className="info-label">Test Step-Acc</span>
                  <span className="info-value">47.8%</span>
                </div>
                <div className="info-row">
                  <span className="info-label">Test Macro-F1</span>
                  <span className="info-value">54.7%</span>
                </div>
                <div className="info-row">
                  <span className="info-label">Trainable Params</span>
                  <span className="info-value">1.42%</span>
                </div>
              </div>
              <div className="link-row">
                <a className="repo-link" href="https://github.com/Minato-sudo/agentsight" target="_blank" rel="noreferrer">
                  ⌥ GitHub
                </a>
                <a className="repo-link" href="https://huggingface.co/talha1234567/Agentic-Ai" target="_blank" rel="noreferrer">
                  🤗 HuggingFace
                </a>
                <a className="repo-link" href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">
                  📄 API Docs
                </a>
              </div>
            </div>
          )}
        </div>

        {/* ── Results column ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

          {/* Verdict */}
          <div className="card">
            <div className="card-title"><span>🔍</span> Detection Result</div>

            {!result && !loading && (
              <div className="empty-state">
                <div className="empty-icon">👁</div>
                <div className="empty-text">Paste a trajectory and click Analyze</div>
              </div>
            )}

            {result && (
              <>
                <div className={`verdict-card ${result.is_hallucinated ? 'hallucinated' : 'clean'}`}>
                  <div className="verdict-icon">{result.is_hallucinated ? '🚨' : '✅'}</div>
                  <div>
                    <div className="verdict-label">
                      {result.is_hallucinated ? 'Hallucination Detected' : 'Trajectory Clean'}
                    </div>
                    <div className="verdict-sub">
                      {result.is_hallucinated
                        ? `Root cause: Step ${result.predicted_root_cause_step} · Confidence: ${(result.max_hallucination_prob * 100).toFixed(1)}%`
                        : `Max probability: ${(result.max_hallucination_prob * 100).toFixed(1)}% (below threshold ${result.threshold})`
                      }
                    </div>
                  </div>
                </div>

                <div className="stat-row">
                  <div className="stat-chip">
                    <div className="stat-value">{result.n_steps}</div>
                    <div className="stat-label">Steps</div>
                  </div>
                  <div className="stat-chip">
                    <div className="stat-value">{(result.max_hallucination_prob * 100).toFixed(0)}%</div>
                    <div className="stat-label">Max Prob</div>
                  </div>
                  <div className="stat-chip">
                    <div className="stat-value">{result.processing_time_ms?.toFixed(0)}ms</div>
                    <div className="stat-label">Latency</div>
                  </div>
                  <div className="stat-chip">
                    <div className="stat-value">{result.threshold}</div>
                    <div className="stat-label">Threshold</div>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Timeline */}
          <div className="card" style={{ flex: 1 }}>
            <div className="card-title"><span>📊</span> Step-by-Step Timeline</div>

            {!result && (
              <div className="empty-state">
                <div className="empty-icon">📊</div>
                <div className="empty-text">Timeline will appear after analysis</div>
              </div>
            )}

            {result && (
              <div className="timeline">
                {result.step_analysis.map((step, idx) => (
                  <div
                    key={step.step}
                    className={`timeline-step ${step.is_flagged ? 'flagged' : ''}`}
                    style={{ animationDelay: `${idx * 0.07}s` }}
                    id={`step-${step.step}`}
                  >
                    <div className="step-num">{step.step}</div>
                    <div className="step-body">
                      <div className="step-meta">
                        <span className="step-title">
                          {step.is_flagged ? '⚠ Flagged — likely root cause' : `Step ${step.step}`}
                        </span>
                        <span className={`prob-badge ${probClass(step.hallucination_probability)}`}>
                          {(step.hallucination_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      {step.content_preview && (
                        <div className="step-preview">
                          {step.content_preview}{step.content_preview.length >= 120 ? '…' : ''}
                        </div>
                      )}
                      <div className="prob-bar-bg">
                        <div
                          className="prob-bar-fill"
                          style={{
                            width: `${step.hallucination_probability * 100}%`,
                            background: probColor(step.hallucination_probability),
                          }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
