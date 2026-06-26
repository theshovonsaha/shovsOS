import React, { useEffect, useMemo, useState } from 'react';

interface HarnessMode {
  id: string;
  label: string;
  included_wedges: string[];
  best_for: string;
  expected_failure: string;
}

interface HarnessWedge {
  id: string;
  label: string;
  plain: string;
  impact: string;
  limitation: string;
  test: string;
}

interface HarnessPaper {
  id: string;
  title: string;
  url: string;
  alignment: string;
}

interface HarnessCatalog {
  title: string;
  subtitle: string;
  workflow: string[];
  modes: HarnessMode[];
  wedges: HarnessWedge[];
  papers: HarnessPaper[];
  benchmark: {
    suite: string;
    endpoint: string;
    deterministic: boolean;
    live_llm_required: boolean;
    network_required: boolean;
  };
  comparison?: {
    suite: string;
    endpoint: string;
    frontend_endpoint?: string;
    modes: string[];
    live_model_required: boolean;
    network_required: boolean;
  };
  runtime?: {
    suite: string;
    endpoint: string;
    frontend_endpoint?: string;
    deterministic: boolean;
    live_model_required: boolean;
    network_required: boolean;
  };
}

interface BenchmarkResultItem {
  scenario: string;
  passed: boolean;
  score: number;
  issues: string[];
  summary: string;
  evidence?: Record<string, unknown>;
}

interface BenchmarkRun {
  suite: string;
  passed: boolean;
  score: number;
  results: BenchmarkResultItem[];
  mode_implications: Array<{
    mode: string;
    label: string;
    coverage: number;
    included_wedges: string[];
  }>;
}

interface RuntimeRun {
  suite: string;
  version: number;
  deterministic: boolean;
  live_model_required: boolean;
  network_required: boolean;
  passed: boolean;
  score: number;
  results: BenchmarkResultItem[];
  coverage?: {
    entrypoints?: string[];
    failure_modes?: string[];
    frontend_contract?: string[];
  };
  operator_guidance?: string[];
}

interface CompareResult {
  mode: string;
  label: string;
  control_policy?: string;
  ledger_mode?: string;
  passed: boolean;
  score: number;
  issues: string[];
  summary: string;
  trace: Array<Record<string, unknown>>;
  trace_summary?: {
    event_count?: number;
    policy_violations?: number;
    tool_events?: number;
    completion_gate?: string;
  };
  state_eval: {
    passed?: boolean;
    detail?: string;
    issues?: string[];
    state?: Record<string, unknown>;
  };
  policy_eval?: {
    passed?: boolean;
    score?: number;
    detail?: string;
    issues?: string[];
    state?: Record<string, unknown>;
  };
}

interface CompareRun {
  suite: string;
  objective: string;
  live_model: boolean;
  deterministic: boolean;
  contract: Record<string, unknown>;
  pass_graph: Record<string, unknown>;
  results: CompareResult[];
  takeaway: string;
}

type HarnessTab = 'compare' | 'runtime' | 'wedges' | 'workflow' | 'benchmarks' | 'papers';

const TAB_LABELS: Record<HarnessTab, string> = {
  compare: 'Compare',
  runtime: 'Runtime',
  wedges: 'Wedges',
  workflow: 'Workflow',
  benchmarks: 'Benchmarks',
  papers: 'Papers',
};

const statusLabel = (passed: boolean) => (passed ? 'pass' : 'attention');
const percent = (value?: number) => `${Math.round((value || 0) * 100)}%`;
const textValue = (value: unknown, fallback = 'not recorded') => {
  const text = String(value ?? '').trim();
  return text || fallback;
};
const traceMatches = (event: Record<string, unknown>, values: string[]) => {
  const phase = String(event.phase || '').toLowerCase();
  const action = String(event.action || '').toLowerCase();
  const type = String(event.event_type || '').toLowerCase();
  const tool = String(event.tool || event.tool_name || '').toLowerCase();
  return values.some((value) => {
    const needle = value.toLowerCase();
    return phase === needle || action === needle || type === needle || tool === needle;
  });
};
const countTrace = (trace: Array<Record<string, unknown>>, values: string[]) =>
  trace.filter((event) => traceMatches(event, values)).length;
const findTrace = (trace: Array<Record<string, unknown>>, values: string[]) =>
  trace.find((event) => traceMatches(event, values));

interface RunMapStep {
  id: string;
  label: string;
  status: 'pass' | 'fail' | 'pending';
  detail: string;
  count?: number;
}

const buildRunMap = (result: CompareResult): RunMapStep[] => {
  const trace = result.trace || [];
  const state = (result.state_eval?.state || {}) as Record<string, unknown>;
  const policy = result.control_policy || findTrace(trace, ['policy'])?.control_policy;
  const entityCount = Array.isArray(state.entities) ? state.entities.length : undefined;
  const searchCount = countTrace(trace, ['web_search']);
  const fetchCount = countTrace(trace, ['web_fetch']);
  const gate = result.trace_summary?.completion_gate || (result.passed ? 'passed' : 'blocked_or_unproven');
  const graphCount = countTrace(trace, ['pass_graph_execution', 'pass_node_started', 'pass_node_completed']);
  const verify = findTrace(trace, ['verify', 'state_eval', 'completion_gate']);
  return [
    {
      id: 'policy',
      label: 'Policy',
      status: policy ? 'pass' : 'pending',
      detail: textValue(policy),
    },
    {
      id: 'contract',
      label: 'Contract',
      status: entityCount ? 'pass' : result.mode.includes('shovs') ? 'pending' : 'fail',
      detail: entityCount ? `${entityCount} locked entities` : 'not recorded',
      count: entityCount,
    },
    {
      id: 'search',
      label: 'Search',
      status: searchCount ? 'pass' : 'pending',
      detail: searchCount ? `${searchCount} search events` : 'not recorded',
      count: searchCount,
    },
    {
      id: 'fetch',
      label: 'Fetch',
      status: fetchCount ? 'pass' : 'pending',
      detail: fetchCount ? `${fetchCount} fetch events` : 'not recorded',
      count: fetchCount,
    },
    {
      id: 'graph',
      label: 'Graph',
      status: result.control_policy === 'graph_harness' ? (graphCount ? 'pass' : 'pending') : 'pending',
      detail: graphCount ? `${graphCount} graph events` : result.control_policy === 'graph_harness' ? 'not recorded' : 'not used',
      count: graphCount,
    },
    {
      id: 'verify',
      label: 'Verify',
      status: verify || result.state_eval?.passed ? 'pass' : 'pending',
      detail: textValue(verify?.summary || result.state_eval?.detail),
    },
    {
      id: 'gate',
      label: 'Gate',
      status: gate === 'passed' ? 'pass' : 'fail',
      detail: gate,
    },
  ];
};

const summarizeEvidenceByEntity = (result: CompareResult) => {
  const state = (result.state_eval?.state || {}) as Record<string, unknown>;
  const fetched = state.fetched_by_entity as Record<string, unknown> | undefined;
  if (!fetched || typeof fetched !== 'object') return [];
  return Object.entries(fetched).map(([entity, urls]) => ({
    entity,
    count: Array.isArray(urls) ? urls.length : 0,
    urls: Array.isArray(urls) ? urls.map((url) => String(url)) : [],
  }));
};

export const HarnessLabView: React.FC = () => {
  const [catalog, setCatalog] = useState<HarnessCatalog | null>(null);
  const [activeTab, setActiveTab] = useState<HarnessTab>('compare');
  const [selectedMode, setSelectedMode] = useState('shovs_enforced');
  const [selectedWedge, setSelectedWedge] = useState<string | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkRun | null>(null);
  const [runtimeRun, setRuntimeRun] = useState<RuntimeRun | null>(null);
  const [compareTask, setCompareTask] = useState('Search top 3 stocks today, then search each, fetch 3 URLs each.');
  const [compareRun, setCompareRun] = useState<CompareRun | null>(null);
  const [selectedCompareMode, setSelectedCompareMode] = useState('shovs_graph_harness');
  const [loadingCatalog, setLoadingCatalog] = useState(true);
  const [runningBenchmark, setRunningBenchmark] = useState(false);
  const [runningRuntime, setRunningRuntime] = useState(false);
  const [runningCompare, setRunningCompare] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoadingCatalog(true);
      setError('');
      try {
        const response = await fetch('/api/harness-lab/catalog');
        if (!response.ok) throw new Error(`catalog failed: ${response.status}`);
        const payload = (await response.json()) as HarnessCatalog;
        if (!cancelled) {
          setCatalog(payload);
          setSelectedWedge(payload.wedges[0]?.id || null);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load Harness Lab.');
      } finally {
        if (!cancelled) setLoadingCatalog(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const wedgeById = useMemo(() => {
    const map = new Map<string, HarnessWedge>();
    catalog?.wedges.forEach((wedge) => map.set(wedge.id, wedge));
    return map;
  }, [catalog]);

  const currentMode = catalog?.modes.find((mode) => mode.id === selectedMode) || catalog?.modes[0];
  const currentWedge = selectedWedge ? wedgeById.get(selectedWedge) : catalog?.wedges[0];
  const selectedCompareResult =
    compareRun?.results.find((item) => item.mode === selectedCompareMode) ||
    compareRun?.results[compareRun.results.length - 1];
  const selectedRunMap = selectedCompareResult ? buildRunMap(selectedCompareResult) : [];
  const selectedEvidenceByEntity = selectedCompareResult ? summarizeEvidenceByEntity(selectedCompareResult) : [];

  const runBenchmark = async () => {
    setRunningBenchmark(true);
    setError('');
    try {
      const response = await fetch('/api/harness-lab/benchmark/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (!response.ok) throw new Error(`benchmark failed: ${response.status}`);
      setBenchmark((await response.json()) as BenchmarkRun);
      setActiveTab('benchmarks');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Benchmark failed.');
    } finally {
      setRunningBenchmark(false);
    }
  };

  const runRuntimeHarness = async () => {
    setRunningRuntime(true);
    setError('');
    try {
      const response = await fetch('/api/harness-lab/runtime/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (!response.ok) throw new Error(`runtime harness failed: ${response.status}`);
      setRuntimeRun((await response.json()) as RuntimeRun);
      setActiveTab('runtime');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Runtime harness failed.');
    } finally {
      setRunningRuntime(false);
    }
  };

  const runCompare = async () => {
    setRunningCompare(true);
    setError('');
    try {
      const response = await fetch('/api/harness-lab/compare/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ objective: compareTask }),
      });
      if (!response.ok) throw new Error(`comparison failed: ${response.status}`);
      const payload = (await response.json()) as CompareRun;
      setCompareRun(payload);
      setSelectedCompareMode(payload.results[payload.results.length - 1]?.mode || 'shovs_graph_harness');
      setActiveTab('compare');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Comparison failed.');
    } finally {
      setRunningCompare(false);
    }
  };

  if (loadingCatalog) {
    return (
      <section className='harness-lab-view'>
        <div className='harness-lab-empty'>Loading harness metadata...</div>
      </section>
    );
  }

  if (!catalog) {
    return (
      <section className='harness-lab-view'>
        <div className='harness-lab-empty error'>{error || 'Harness Lab is not available.'}</div>
      </section>
    );
  }

  return (
    <section className='harness-lab-view'>
      <div className='harness-lab-header'>
        <div>
          <div className='workspace-card-kicker'>Runtime Evidence</div>
          <h2>{catalog.title}</h2>
          <p>{catalog.subtitle}</p>
        </div>
        <button className='shovs-send-btn' onClick={runBenchmark} disabled={runningBenchmark}>
          {runningBenchmark ? 'Running...' : 'Run Core Benchmark'}
        </button>
        <button className='shovs-send-btn secondary' onClick={runRuntimeHarness} disabled={runningRuntime}>
          {runningRuntime ? 'Running...' : 'Run Runtime Harness'}
        </button>
      </div>

      <div className='harness-tabs' role='tablist' aria-label='Harness Lab sections'>
        {(Object.keys(TAB_LABELS) as HarnessTab[]).map((tab) => (
          <button
            key={tab}
            className={activeTab === tab ? 'active' : ''}
            onClick={() => setActiveTab(tab)}
          >
            {TAB_LABELS[tab]}
          </button>
        ))}
      </div>

      {error ? <div className='harness-lab-error'>{error}</div> : null}

      {activeTab === 'compare' ? (
        <>
          <div className='harness-run-box'>
            <div>
              <label htmlFor='harness-compare-task'>Task to compare</label>
              <textarea
                id='harness-compare-task'
                value={compareTask}
                onChange={(event) => setCompareTask(event.target.value)}
              />
            </div>
            <button className='shovs-send-btn' onClick={runCompare} disabled={runningCompare}>
              {runningCompare ? 'Running...' : 'Run Mode Comparison'}
            </button>
          </div>

          <div className='harness-compare-grid'>
            <div className='harness-mode-list'>
              {compareRun ? (
                compareRun.results.map((item) => (
                  <button
                    key={item.mode}
                    className={`harness-mode-card ${selectedCompareMode === item.mode ? 'active' : ''}`}
                    onClick={() => setSelectedCompareMode(item.mode)}
                  >
                    <span>{item.label}</span>
                    <small>{percent(item.score)} state · {percent(item.policy_eval?.score)} policy</small>
                  </button>
                ))
              ) : (
                catalog.modes.map((mode) => (
                  <button
                    key={mode.id}
                    className={`harness-mode-card ${selectedMode === mode.id ? 'active' : ''}`}
                    onClick={() => setSelectedMode(mode.id)}
                  >
                    <span>{mode.label}</span>
                    <small>{mode.included_wedges.length} wedge{mode.included_wedges.length === 1 ? '' : 's'}</small>
                  </button>
                ))
              )}
            </div>
            {compareRun && selectedCompareResult ? (
              <div className='harness-focus-panel'>
                <div className='harness-panel-title'>{selectedCompareResult.label}</div>
                <div className='harness-compare-score-row'>
                  <div className={selectedCompareResult.passed ? 'pass' : 'fail'}>
                    <span>State score</span>
                    <strong>{percent(selectedCompareResult.score)}</strong>
                  </div>
                  <div className={selectedCompareResult.policy_eval?.passed ? 'pass' : 'fail'}>
                    <span>Policy score</span>
                    <strong>{percent(selectedCompareResult.policy_eval?.score)}</strong>
                  </div>
                  <div>
                    <span>Policy</span>
                    <strong>{selectedCompareResult.control_policy || 'not recorded'}</strong>
                  </div>
                  <div>
                    <span>Gate</span>
                    <strong>{selectedCompareResult.trace_summary?.completion_gate || (selectedCompareResult.passed ? 'passed' : 'blocked')}</strong>
                  </div>
                </div>
                <div className='harness-trace-summary'>
                  <div>
                    <span>Ledger</span>
                    <strong>{selectedCompareResult.ledger_mode || 'not recorded'}</strong>
                  </div>
                  <div>
                    <span>Events</span>
                    <strong>{selectedCompareResult.trace_summary?.event_count ?? selectedCompareResult.trace.length}</strong>
                  </div>
                  <div>
                    <span>Tools</span>
                    <strong>{selectedCompareResult.trace_summary?.tool_events ?? 'not recorded'}</strong>
                  </div>
                  <div>
                    <span>Violations</span>
                    <strong>{selectedCompareResult.trace_summary?.policy_violations ?? 'not recorded'}</strong>
                  </div>
                </div>
                <div className='harness-run-map'>
                  {selectedRunMap.map((step) => (
                    <div key={step.id} className={`harness-run-map-step ${step.status}`}>
                      <span>{step.label}</span>
                      <strong>{step.count ?? (step.status === 'pass' ? 'ok' : step.status)}</strong>
                      <p>{step.detail}</p>
                    </div>
                  ))}
                </div>
                {selectedEvidenceByEntity.length ? (
                  <div className='harness-evidence-strip'>
                    {selectedEvidenceByEntity.map((item) => (
                      <div key={item.entity}>
                        <span>{item.entity}</span>
                        <strong>{item.count} source{item.count === 1 ? '' : 's'}</strong>
                        <small>{item.urls.slice(0, 2).join(' · ') || 'not recorded'}</small>
                      </div>
                    ))}
                  </div>
                ) : null}
                <div className='harness-panel-row'>
                  <span>Summary</span>
                  <p>{selectedCompareResult.summary}</p>
                </div>
                <div className='harness-panel-row'>
                  <span>Issues</span>
                  <p>{selectedCompareResult.issues.length ? selectedCompareResult.issues.join(' · ') : 'none'}</p>
                </div>
                <div className='harness-panel-row'>
                  <span>Policy eval</span>
                  <p>{selectedCompareResult.policy_eval?.detail || 'not recorded'}</p>
                </div>
                <div className='harness-policy-box'>
                  <div>
                    <span>Expected policy proof</span>
                    <strong>{selectedCompareResult.policy_eval?.passed ? 'present' : 'missing or partial'}</strong>
                  </div>
                  <div>
                    <span>State proof</span>
                    <strong>{selectedCompareResult.state_eval?.passed ? 'matched' : 'not matched'}</strong>
                  </div>
                  <div>
                    <span>Unsupported issues</span>
                    <strong>{selectedCompareResult.policy_eval?.issues?.length || selectedCompareResult.issues.length || 0}</strong>
                  </div>
                </div>
                <div className='harness-trace-list'>
                  {selectedCompareResult.trace.map((event, index) => (
                    <div key={`${selectedCompareResult.mode}-${index}`} className='harness-trace-row'>
                      <span>{String(event.phase || event.tool || event.action || 'event')}</span>
                      <strong>{String(event.action || event.tool || event.actor || 'not recorded')}</strong>
                      <p>{String(event.summary || event.query || event.url || 'not recorded')}</p>
                    </div>
                  ))}
                </div>
                <details className='harness-json-details'>
                  <summary>Contract and state eval</summary>
                  <pre>{JSON.stringify({
                    contract: compareRun.contract,
                    pass_graph: compareRun.pass_graph,
                    state_eval: selectedCompareResult.state_eval,
                    policy_eval: selectedCompareResult.policy_eval,
                    trace_summary: selectedCompareResult.trace_summary,
                  }, null, 2)}</pre>
                </details>
              </div>
            ) : currentMode ? (
              <div className='harness-focus-panel'>
                <div className='harness-panel-title'>{currentMode.label}</div>
                <div className='harness-panel-row'>
                  <span>Best for</span>
                  <p>{currentMode.best_for}</p>
                </div>
                <div className='harness-panel-row'>
                  <span>Known limit</span>
                  <p>{currentMode.expected_failure}</p>
                </div>
                <div className='harness-wedge-chip-row'>
                  {catalog.wedges.map((wedge) => {
                    const included = currentMode.included_wedges.includes(wedge.id);
                    return (
                      <span key={wedge.id} className={`harness-wedge-chip ${included ? 'on' : 'off'}`}>
                        {included ? '✓' : '–'} {wedge.label}
                      </span>
                    );
                  })}
                </div>
              </div>
            ) : null}
          </div>
          {compareRun ? <div className='harness-takeaway'>{compareRun.takeaway}</div> : null}
        </>
      ) : null}

      {activeTab === 'wedges' ? (
        <div className='harness-compare-grid'>
          <div className='harness-mode-list'>
            {catalog.wedges.map((wedge) => (
              <button
                key={wedge.id}
                className={`harness-mode-card ${selectedWedge === wedge.id ? 'active' : ''}`}
                onClick={() => setSelectedWedge(wedge.id)}
              >
                <span>{wedge.label}</span>
                <small>{wedge.test}</small>
              </button>
            ))}
          </div>
          {currentWedge ? (
            <div className='harness-focus-panel'>
              <div className='harness-panel-title'>{currentWedge.label}</div>
              <div className='harness-panel-row'>
                <span>Plain meaning</span>
                <p>{currentWedge.plain}</p>
              </div>
              <div className='harness-panel-row'>
                <span>Impact</span>
                <p>{currentWedge.impact}</p>
              </div>
              <div className='harness-panel-row'>
                <span>Limitation</span>
                <p>{currentWedge.limitation}</p>
              </div>
              <div className='harness-test-pill'>Test: {currentWedge.test}</div>
            </div>
          ) : null}
        </div>
      ) : null}

      {activeTab === 'runtime' ? (
        <div className='harness-benchmark-panel'>
          <div className='harness-benchmark-summary'>
            <div>
              <span>Suite</span>
              <strong>{runtimeRun?.suite || catalog.runtime?.suite || 'shovs_runtime_harness'}</strong>
            </div>
            <div>
              <span>Status</span>
              <strong className={runtimeRun?.passed ? 'pass' : ''}>
                {runtimeRun ? statusLabel(runtimeRun.passed) : 'not run'}
              </strong>
            </div>
            <div>
              <span>Score</span>
              <strong>{runtimeRun ? `${Math.round(runtimeRun.score * 100)}%` : 'not recorded'}</strong>
            </div>
            <div>
              <span>Live model</span>
              <strong>{catalog.runtime?.live_model_required ? 'required' : 'not required'}</strong>
            </div>
          </div>
          {!runtimeRun ? (
            <div className='harness-lab-empty'>
              Run the runtime harness to execute isolated RunEngine probes with fake small-model outputs and deterministic tools.
            </div>
          ) : (
            <>
              <div className='harness-policy-box'>
                <div>
                  <span>Entrypoints</span>
                  <strong>{runtimeRun.coverage?.entrypoints?.join(' · ') || 'not recorded'}</strong>
                </div>
                <div>
                  <span>Failure modes</span>
                  <strong>{runtimeRun.coverage?.failure_modes?.join(' · ') || 'not recorded'}</strong>
                </div>
                <div>
                  <span>Frontend contract</span>
                  <strong>{runtimeRun.coverage?.frontend_contract?.join(' · ') || 'not recorded'}</strong>
                </div>
              </div>
              <div className='harness-result-list'>
                {runtimeRun.results.map((item) => (
                  <div key={item.scenario} className={`harness-result-card ${item.passed ? 'pass' : 'fail'}`}>
                    <div>
                      <strong>{item.scenario.replace(/_/g, ' ')}</strong>
                      <p>{item.summary}</p>
                      {item.issues.length ? <small>{item.issues.join(' · ')}</small> : null}
                    </div>
                    <span>{Math.round(item.score * 100)}%</span>
                  </div>
                ))}
              </div>
              {runtimeRun.operator_guidance?.length ? (
                <div className='harness-takeaway'>
                  {runtimeRun.operator_guidance.join(' ')}
                </div>
              ) : null}
              <details className='harness-json-details'>
                <summary>Runtime harness payload</summary>
                <pre>{JSON.stringify(runtimeRun, null, 2)}</pre>
              </details>
            </>
          )}
        </div>
      ) : null}

      {activeTab === 'workflow' ? (
        <div className='harness-workflow'>
          {catalog.workflow.map((step, index) => (
            <div key={step} className='harness-workflow-step'>
              <span>{String(index + 1).padStart(2, '0')}</span>
              <strong>{step}</strong>
            </div>
          ))}
        </div>
      ) : null}

      {activeTab === 'benchmarks' ? (
        <div className='harness-benchmark-panel'>
          <div className='harness-benchmark-summary'>
            <div>
              <span>Suite</span>
              <strong>{benchmark?.suite || catalog.benchmark.suite}</strong>
            </div>
            <div>
              <span>Status</span>
              <strong className={benchmark?.passed ? 'pass' : ''}>
                {benchmark ? statusLabel(benchmark.passed) : 'not run'}
              </strong>
            </div>
            <div>
              <span>Score</span>
              <strong>{benchmark ? `${Math.round(benchmark.score * 100)}%` : 'not recorded'}</strong>
            </div>
            <div>
              <span>Live LLM</span>
              <strong>{catalog.benchmark.live_llm_required ? 'required' : 'not required'}</strong>
            </div>
          </div>
          {!benchmark ? (
            <div className='harness-lab-empty'>Run the benchmark to see deterministic results.</div>
          ) : (
            <div className='harness-result-list'>
              {benchmark.results.map((item) => (
                <div key={item.scenario} className={`harness-result-card ${item.passed ? 'pass' : 'fail'}`}>
                  <div>
                    <strong>{item.scenario.replace(/_/g, ' ')}</strong>
                    <p>{item.summary}</p>
                  </div>
                  <span>{Math.round(item.score * 100)}%</span>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : null}

      {activeTab === 'papers' ? (
        <div className='harness-paper-grid'>
          {catalog.papers.map((paper) => (
            <a key={paper.id} className='harness-paper-card' href={paper.url} target='_blank' rel='noreferrer'>
              <strong>{paper.title}</strong>
              <p>{paper.alignment}</p>
              <span>{paper.url.replace('https://', '')}</span>
            </a>
          ))}
        </div>
      ) : null}
    </section>
  );
};
