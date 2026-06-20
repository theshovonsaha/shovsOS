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

type HarnessTab = 'compare' | 'wedges' | 'workflow' | 'benchmarks' | 'papers';

const TAB_LABELS: Record<HarnessTab, string> = {
  compare: 'Compare',
  wedges: 'Wedges',
  workflow: 'Workflow',
  benchmarks: 'Benchmarks',
  papers: 'Papers',
};

const statusLabel = (passed: boolean) => (passed ? 'pass' : 'attention');

export const HarnessLabView: React.FC = () => {
  const [catalog, setCatalog] = useState<HarnessCatalog | null>(null);
  const [activeTab, setActiveTab] = useState<HarnessTab>('compare');
  const [selectedMode, setSelectedMode] = useState('shovs_enforced');
  const [selectedWedge, setSelectedWedge] = useState<string | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkRun | null>(null);
  const [loadingCatalog, setLoadingCatalog] = useState(true);
  const [runningBenchmark, setRunningBenchmark] = useState(false);
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
        <div className='harness-compare-grid'>
          <div className='harness-mode-list'>
            {catalog.modes.map((mode) => (
              <button
                key={mode.id}
                className={`harness-mode-card ${selectedMode === mode.id ? 'active' : ''}`}
                onClick={() => setSelectedMode(mode.id)}
              >
                <span>{mode.label}</span>
                <small>{mode.included_wedges.length} wedge{mode.included_wedges.length === 1 ? '' : 's'}</small>
              </button>
            ))}
          </div>
          {currentMode ? (
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

