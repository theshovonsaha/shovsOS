import React, { useState, useEffect, useCallback } from 'react';
import { PremiumSelect } from './PremiumSelect';
import './OptionsPanel.css';
import { withOwnerPayload, withOwnerQuery } from '../owner';

interface Memory {
  id: number;
  subject: string;
  predicate: string;
  object: string;
  created_at: string;
  similarity?: number;
}

interface MemoryStateFact {
  subject: string;
  predicate: string;
  object: string;
  status: 'current' | 'superseded';
  valid_from?: number | null;
  valid_to?: number | null;
  created_at?: string | null;
}

interface MemorySignal {
  event_type: string;
  label: string;
  summary: string;
  created_at?: string;
}

interface SessionMemoryState {
  summary: {
    deterministic_fact_count: number;
    superseded_fact_count: number;
    candidate_signal_count: number;
    stance_signal_count: number;
    context_line_count: number;
    memory_signal_count: number;
  };
  deterministic_facts: MemoryStateFact[];
  superseded_facts: MemoryStateFact[];
  candidate_signals: Array<{
    text: string;
    reason: string;
    signal_type?: string;
    topic?: string;
    confidence?: string;
    superseded?: boolean;
  }>;
  stance_signals: Array<{
    text: string;
    reason: string;
    signal_type?: string;
    topic?: string;
    confidence?: string;
    superseded?: boolean;
  }>;
  context_preview: string[];
  recent_memory_signals: MemorySignal[];
  explanation: string[];
}

interface MemoryBenchmarkResult {
  ran_at: string;
  overall_score: number;
  metrics: {
    deterministic_extraction: {
      precision: number;
      recall: number;
      f1: number;
      void_accuracy: number;
      duration_ms: number;
    };
    direct_fact_guard: {
      accuracy: number;
      duration_ms: number;
    };
    semantic_retrieval: {
      hit_rate_at_3: number;
      mrr_at_3: number;
      duration_ms: number;
    };
  };
}

interface OptionsPanelProps {
  sessionId: string | null;
  contextLines: number;
  currentSearchEngine: string;
  setCurrentSearchEngine: (engine: string) => void;
  models: Record<string, string[]>;
  runtimePath: 'legacy' | 'run_engine';
  setRuntimePath: (val: 'legacy' | 'run_engine') => void;
  usePlanner: boolean;
  setUsePlanner: (val: boolean) => void;
  loopMode: 'auto' | 'single' | 'managed';
  setLoopMode: (val: 'auto' | 'single' | 'managed') => void;
  maxToolCalls: string;
  setMaxToolCalls: (val: string) => void;
  maxTurns: string;
  setMaxTurns: (val: string) => void;
  plannerModel: string;
  setPlannerModel: (val: string) => void;
  contextModel: string;
  setContextModel: (val: string) => void;
  embedModel: string;
  setEmbedModel: (val: string) => void;
  embedModels: Record<string, string[]>;
  contextMode: 'v1' | 'v2' | 'v3';
  setSessionContextMode: (mode: 'v1' | 'v2' | 'v3') => void;
  clearSessionContext: () => void;
  showPlannerLog: boolean;
  setShowPlannerLog: (val: boolean) => void;
  showActorThought: boolean;
  setShowActorThought: (val: boolean) => void;
  showObserverActivity: boolean;
  setShowObserverActivity: (val: boolean) => void;
}

interface StorageStoreInfo {
  key: string;
  path: string;
  exists: boolean;
  kind: string;
  size_bytes: number;
  records?: number | null;
}

interface StorageStatus {
  generated_at: string;
  backup_root: string;
  stores: Record<string, StorageStoreInfo>;
}

interface BackupInfo {
  name: string;
  path: string;
  created_at?: string;
  stores: string[];
  size_bytes: number;
}

export const OptionsPanel: React.FC<OptionsPanelProps> = ({
  sessionId,
  contextLines,
  currentSearchEngine,
  setCurrentSearchEngine,
  models,
  runtimePath,
  setRuntimePath,
  usePlanner,
  setUsePlanner,
  loopMode,
  setLoopMode,
  maxToolCalls,
  setMaxToolCalls,
  maxTurns,
  setMaxTurns,
  plannerModel,
  setPlannerModel,
  contextModel,
  setContextModel,
  embedModel,
  setEmbedModel,
  contextMode,
  setSessionContextMode,
  clearSessionContext,
  showPlannerLog,
  setShowPlannerLog,
  showActorThought,
  setShowActorThought,
  showObserverActivity,
  setShowObserverActivity,
  embedModels,
}) => {
  const loopModes: Array<{
    value: 'auto' | 'single' | 'managed';
    label: string;
  }> = [
    { value: 'auto', label: 'Auto' },
    { value: 'single', label: 'Single' },
    { value: 'managed', label: 'Managed' },
  ];
  const effectiveLoopHint =
    loopMode === 'managed'
      ? 'Current selection: Managed.'
      : loopMode === 'single'
        ? 'Current selection: Single.'
        : usePlanner
          ? 'Auto prefers managed for non-trivial turns, but local runtimes may downgrade to single.'
          : 'Auto resolves to single unless you re-enable Manager Agent.';

  const nextContextMode =
    contextMode === 'v1' ? 'v2' : contextMode === 'v2' ? 'v3' : 'v1';
  const [memories, setMemories] = useState<Memory[]>([]);
  const [total, setTotal] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Memory[] | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionContext, setSessionContext] = useState<string[]>([]);
  const [sessionMemoryState, setSessionMemoryState] =
    useState<SessionMemoryState | null>(null);
  const [activeTab, setActiveTab] = useState<'settings' | 'memory' | 'context'>(
    'settings',
  );
  const [confirmClear, setConfirmClear] = useState(false);
  const [benchmarkResult, setBenchmarkResult] =
    useState<MemoryBenchmarkResult | null>(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [benchmarkError, setBenchmarkError] = useState('');
  const [storageStatus, setStorageStatus] = useState<StorageStatus | null>(
    null,
  );
  const [storageBackups, setStorageBackups] = useState<BackupInfo[]>([]);
  const [storageLoading, setStorageLoading] = useState(false);
  const [storageMessage, setStorageMessage] = useState('');
  const [backupLabel, setBackupLabel] = useState('');
  const [backupFirst, setBackupFirst] = useState(true);
  const [preserveDefaultAgent, setPreserveDefaultAgent] = useState(true);
  const [storageSelection, setStorageSelection] = useState({
    sessions: true,
    agents: false,
    semantic_memory: true,
    tool_results: true,
    vector_memory: true,
    session_rag: true,
  });

  const loadMemories = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await fetch(withOwnerQuery('/api/memory')).then((r) =>
        r.json(),
      );
      setMemories(data.memories || []);
      setTotal(data.total || 0);
    } catch (e) {
      console.error('Failed to load memories', e);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const loadContext = useCallback(async () => {
    if (!sessionId) return;
    try {
      const data = await fetch(
        withOwnerQuery(`/api/sessions/${sessionId}/context`),
      ).then((r) => r.json());
      setSessionContext(data.context || []);
    } catch (e) {
      console.error('Failed to load context', e);
    }
  }, [sessionId]);

  const loadMemoryState = useCallback(async () => {
    if (!sessionId) {
      setSessionMemoryState(null);
      return;
    }
    try {
      const data = await fetch(
        withOwnerQuery(`/api/sessions/${sessionId}/memory-state`),
      ).then((r) => r.json());
      setSessionMemoryState(data);
    } catch (e) {
      console.error('Failed to load session memory state', e);
    }
  }, [sessionId]);

  useEffect(() => {
    loadMemories();
  }, [loadMemories]);

  useEffect(() => {
    if (activeTab === 'context') loadContext();
  }, [activeTab, loadContext]);

  useEffect(() => {
    if (activeTab === 'context') loadMemoryState();
  }, [activeTab, loadMemoryState]);

  const loadMemoryBenchmarkLatest = useCallback(async () => {
    try {
      const payload = await fetch(
        withOwnerQuery('/api/memory/benchmark/latest'),
      ).then((r) => r.json());
      setBenchmarkResult(payload.result || null);
    } catch (e) {
      console.error('Failed to load memory benchmark', e);
    }
  }, []);

  useEffect(() => {
    if (activeTab === 'memory') {
      loadMemoryBenchmarkLatest();
    }
  }, [activeTab, loadMemoryBenchmarkLatest]);

  const loadStorage = useCallback(async () => {
    setStorageLoading(true);
    try {
      const [status, backups] = await Promise.all([
        fetch('/api/storage/status').then((r) => r.json()),
        fetch('/api/storage/backups?limit=8').then((r) => r.json()),
      ]);
      setStorageStatus(status);
      setStorageBackups(backups.backups || []);
    } catch (e) {
      console.error('Failed to load storage status', e);
    } finally {
      setStorageLoading(false);
    }
  }, []);

  useEffect(() => {
    if (activeTab === 'settings') {
      loadStorage();
    }
  }, [activeTab, loadStorage]);

  const deleteMemory = async (id: number) => {
    try {
      await fetch(withOwnerQuery(`/api/memory/${id}`), { method: 'DELETE' });
      setMemories((prev) => prev.filter((m) => m.id !== id));
      setSearchResults((prev) =>
        prev ? prev.filter((m) => m.id !== id) : null,
      );
      setTotal((prev) => Math.max(0, prev - 1));
    } catch (e) {
      console.error('Failed to delete memory', e);
    }
  };

  const clearAllMemories = async () => {
    if (!confirmClear) {
      setConfirmClear(true);
      setTimeout(() => setConfirmClear(false), 3000);
      return;
    }
    try {
      await fetch(withOwnerQuery('/api/memory'), { method: 'DELETE' });
      setMemories([]);
      setTotal(0);
      setSearchResults(null);
      setConfirmClear(false);
    } catch (e) {
      console.error('Failed to clear memories', e);
    }
  };

  const searchMemory = async () => {
    if (!searchQuery.trim()) {
      setSearchResults(null);
      return;
    }
    setIsSearching(true);
    try {
      const data = await fetch('/api/memory/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(
          withOwnerPayload({ query: searchQuery, top_k: 10 }),
        ),
      }).then((r) => r.json());
      setSearchResults(data.results || []);
    } catch (e) {
      console.error('Failed to search memories', e);
    } finally {
      setIsSearching(false);
    }
  };

  const runMemoryBenchmark = async () => {
    setBenchmarkLoading(true);
    setBenchmarkError('');
    try {
      const payload = await fetch('/api/memory/benchmark/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(withOwnerPayload({})),
      }).then((r) => r.json());
      setBenchmarkResult(payload);
    } catch (e) {
      console.error('Failed to run memory benchmark', e);
      setBenchmarkError('Benchmark failed.');
    } finally {
      setBenchmarkLoading(false);
    }
  };

  const updateStoreSelection = (key: keyof typeof storageSelection) => {
    setStorageSelection((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const createBackup = async () => {
    setStorageLoading(true);
    setStorageMessage('');
    try {
      const resp = await fetch('/api/storage/backup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...storageSelection,
          backup_label: backupLabel,
        }),
      }).then((r) => r.json());
      setStorageMessage(`Backup created: ${resp.backup_name}`);
      await loadStorage();
    } catch (e) {
      console.error('Failed to create backup', e);
      setStorageMessage('Backup failed.');
    } finally {
      setStorageLoading(false);
    }
  };

  const resetStorage = async () => {
    const enabledStores = Object.entries(storageSelection)
      .filter(([, enabled]) => enabled)
      .map(([key]) => key);
    if (enabledStores.length === 0) {
      setStorageMessage('Select at least one store.');
      return;
    }
    if (
      !window.confirm(
        `Reset selected storage now?\n\n${enabledStores.join(', ')}\n\nThis can wipe persistent memory and session state.`,
      )
    ) {
      return;
    }

    setStorageLoading(true);
    setStorageMessage('');
    try {
      const resp = await fetch('/api/storage/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...storageSelection,
          backup_first: backupFirst,
          backup_label: backupLabel,
          preserve_default_agent: preserveDefaultAgent,
        }),
      }).then((r) => r.json());
      setStorageMessage(
        `Reset complete${resp.backup?.backup_name ? ` with backup ${resp.backup.backup_name}` : ''}.`,
      );
      await loadStorage();
    } catch (e) {
      console.error('Failed to reset storage', e);
      setStorageMessage('Reset failed.');
    } finally {
      setStorageLoading(false);
    }
  };

  const displayMemories = searchResults || memories;

  return (
    <div className='options-panel'>
      <div className='options-tabs'>
        <button
          className={`options-tab ${activeTab === 'settings' ? 'active' : ''}`}
          onClick={() => setActiveTab('settings')}
        >
          ⚙ Settings
        </button>
        <button
          className={`options-tab ${activeTab === 'memory' ? 'active' : ''}`}
          onClick={() => setActiveTab('memory')}
        >
          🧠 Memory
          {total > 0 && <span className='tab-badge'>{total}</span>}
        </button>
        <button
          className={`options-tab ${activeTab === 'context' ? 'active' : ''}`}
          onClick={() => setActiveTab('context')}
        >
          📋 Context
          {contextLines > 0 && (
            <span className='tab-badge'>{contextLines}</span>
          )}
        </button>
      </div>

      {activeTab === 'settings' && (
        <div className='options-section'>
          <div
            className='settings-card'
            style={{
              marginBottom: '20px',
              padding: '12px',
              background: 'var(--bg-card)',
              borderRadius: '8px',
              border: '1px solid var(--border-color)',
            }}
          >
            <label
              className='settings-label'
              style={{
                display: 'block',
                marginBottom: '8px',
                fontSize: '13px',
                fontWeight: 600,
              }}
            >
              Web Search Engine
            </label>
            <PremiumSelect
              label=''
              value={currentSearchEngine}
              options={{
                all: [
                  'auto',
                  'duckduckgo',
                  'tavily',
                  'brave',
                  'searxng',
                  'exa',
                ],
              }}
              onChange={(val) => setCurrentSearchEngine(val)}
            />
            <p
              className='settings-help'
              style={{
                marginTop: '8px',
                fontSize: '11px',
                color: 'var(--text-dim)',
                lineHeight: 1.4,
              }}
            >
              Select the engine used by the web_search tool. Auto tries SearXNG
              → Brave → Tavily → Exa → Groq. Set API keys in backend .env for
              key-based providers.
            </p>
          </div>

          <div
            className='settings-card'
            style={{
              marginBottom: '20px',
              padding: '12px',
              background: 'var(--bg-card)',
              borderRadius: '8px',
              border: '1px solid var(--border-color)',
            }}
          >
            <label
              className='settings-label'
              style={{
                display: 'block',
                marginBottom: '8px',
                fontSize: '13px',
                fontWeight: 600,
              }}
            >
              Runtime Path
            </label>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                gap: '8px',
              }}
            >
              {[
                { value: 'run_engine', label: 'Run Engine' },
                { value: 'legacy', label: 'Legacy Core' },
              ].map((option) => {
                const active = runtimePath === option.value;
                return (
                  <button
                    key={option.value}
                    type='button'
                    onClick={() =>
                      setRuntimePath(option.value as 'legacy' | 'run_engine')
                    }
                    style={{
                      minHeight: '42px',
                      borderRadius: '12px',
                      border: active
                        ? '1px solid var(--accent)'
                        : '1px solid var(--border)',
                      background: active
                        ? 'rgba(111, 210, 255, 0.12)'
                        : 'var(--bg)',
                      color: active ? 'var(--accent)' : 'var(--text-dim)',
                      fontSize: '12px',
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                  >
                    {option.label}
                  </button>
                );
              })}
            </div>
            <p
              className='settings-help'
              style={{
                marginTop: '8px',
                fontSize: '11px',
                color: 'var(--text-dim)',
                lineHeight: 1.4,
              }}
            >
              Run Engine uses the new checkpoint-first orchestration path.
              Legacy Core keeps the older mixed runtime for comparison.
            </p>
          </div>

          <div
            className='settings-card'
            style={{
              marginBottom: '20px',
              padding: '12px',
              background: 'var(--surface2)',
              borderRadius: '8px',
              border: '1px solid var(--border)',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <label
                className='settings-label'
                style={{ fontSize: '13px', fontWeight: 600 }}
              >
                Manager Agent (Orchestration)
              </label>
              <input
                type='checkbox'
                checked={usePlanner}
                onChange={(e) => setUsePlanner(e.target.checked)}
                style={{ transform: 'scale(1.2)', cursor: 'pointer' }}
              />
            </div>
            <p
              className='settings-help'
              style={{
                marginTop: '8px',
                fontSize: '11px',
                color: 'var(--text-dim)',
                lineHeight: 1.4,
              }}
            >
              When enabled, the agent uses a specialized "Planner" layer to
              proactively select tools before execution.
            </p>

            {usePlanner && (
              <div style={{ marginTop: '12px' }}>
                <label
                  className='settings-label'
                  style={{
                    display: 'block',
                    marginBottom: '6px',
                    fontSize: '12px',
                  }}
                >
                  Planner Model
                </label>
                <PremiumSelect
                  label=''
                  value={plannerModel || ''}
                  options={{ system: [''], ...models }}
                  onChange={(val) => setPlannerModel(val)}
                />
              </div>
            )}

            <div style={{ marginTop: '12px' }}>
              <label
                className='settings-label'
                style={{
                  display: 'block',
                  marginBottom: '6px',
                  fontSize: '12px',
                }}
              >
                Execution Loop
              </label>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
                  gap: '8px',
                }}
              >
                {loopModes.map((mode) => {
                  const active = loopMode === mode.value;
                  return (
                    <button
                      key={mode.value}
                      type='button'
                      onClick={() => setLoopMode(mode.value)}
                      style={{
                        minHeight: '42px',
                        borderRadius: '12px',
                        border: active
                          ? '1px solid var(--accent)'
                          : '1px solid var(--border)',
                        background: active
                          ? 'rgba(111, 210, 255, 0.12)'
                          : 'var(--bg)',
                        color: active ? 'var(--accent)' : 'var(--text-dim)',
                        fontSize: '12px',
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      {mode.label}
                    </button>
                  );
                })}
              </div>
              <p
                className='settings-help'
                style={{
                  marginTop: '8px',
                  fontSize: '11px',
                  color: 'var(--text-dim)',
                  lineHeight: 1.4,
                }}
              >
                Auto uses the managed controller when planning is enabled.
                Single runs the actor loop directly. Managed forces plan → act →
                observe → verify → commit inside one run.
              </p>
              <p
                className='settings-help'
                style={{
                  marginTop: '6px',
                  fontSize: '11px',
                  color: 'var(--accent)',
                  lineHeight: 1.4,
                }}
              >
                {effectiveLoopHint}
              </p>
            </div>

            <div
              style={{
                marginTop: '12px',
                display: 'grid',
                gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                gap: '10px',
              }}
            >
              <div>
                <label
                  className='settings-label'
                  style={{
                    display: 'block',
                    marginBottom: '6px',
                    fontSize: '12px',
                  }}
                >
                  Max Tool Calls
                </label>
                <input
                  type='number'
                  min={1}
                  max={24}
                  inputMode='numeric'
                  value={maxToolCalls}
                  onChange={(e) =>
                    setMaxToolCalls(
                      e.target.value.replace(/[^\d]/g, '').slice(0, 2),
                    )
                  }
                  placeholder='auto'
                  style={{
                    width: '100%',
                    minHeight: '42px',
                    borderRadius: '10px',
                    border: '1px solid var(--border)',
                    background: 'var(--bg)',
                    color: 'var(--text)',
                    padding: '0 12px',
                    fontSize: '12px',
                  }}
                />
              </div>
              <div>
                <label
                  className='settings-label'
                  style={{
                    display: 'block',
                    marginBottom: '6px',
                    fontSize: '12px',
                  }}
                >
                  Max Turns
                </label>
                <input
                  type='number'
                  min={2}
                  max={12}
                  inputMode='numeric'
                  value={maxTurns}
                  onChange={(e) =>
                    setMaxTurns(
                      e.target.value.replace(/[^\d]/g, '').slice(0, 2),
                    )
                  }
                  placeholder='auto'
                  style={{
                    width: '100%',
                    minHeight: '42px',
                    borderRadius: '10px',
                    border: '1px solid var(--border)',
                    background: 'var(--bg)',
                    color: 'var(--text)',
                    padding: '0 12px',
                    fontSize: '12px',
                  }}
                />
              </div>
            </div>
            <p
              className='settings-help'
              style={{
                marginTop: '8px',
                fontSize: '11px',
                color: 'var(--text-dim)',
                lineHeight: 1.4,
              }}
            >
              Leave blank for adaptive limits. Max turns is total model passes
              in the loop, so values below 2 are not used.
            </p>
          </div>

          <div
            className='settings-card'
            style={{
              marginBottom: '20px',
              padding: '12px',
              background: 'var(--bg-card)',
              borderRadius: '8px',
              border: '1px solid var(--border-color)',
            }}
          >
            <label
              className='settings-label'
              style={{
                display: 'block',
                marginBottom: '8px',
                fontSize: '13px',
                fontWeight: 600,
              }}
            >
              Context Engine Mode
            </label>
            <button
              className='memory-refresh-btn'
              style={{ width: '100%' }}
              onClick={() => setSessionContextMode(nextContextMode)}
            >
              {contextMode === 'v1'
                ? '📋 V1 Linear'
                : contextMode === 'v2'
                  ? '⚡ V2 Convergent'
                  : '🧠 V3 Hybrid'}
            </button>
            <p
              className='settings-help'
              style={{
                marginTop: '8px',
                fontSize: '11px',
                color: 'var(--text-dim)',
                lineHeight: 1.4,
              }}
            >
              Per-session toggle. V1 stores linear durable memory. V2 ranks
              convergent modules by active goals. V3 combines both as an
              experimental hybrid.
            </p>
          </div>

          <div
            className='settings-card'
            style={{
              marginBottom: '20px',
              padding: '12px',
              background: 'var(--bg-card)',
              borderRadius: '8px',
              border: '1px solid var(--border-color)',
            }}
          >
            <label
              className='settings-label'
              style={{
                display: 'block',
                marginBottom: '8px',
                fontSize: '13px',
                fontWeight: 600,
              }}
            >
              Context Engine Intelligence
            </label>
            <PremiumSelect
              label=''
              value={contextModel}
              options={models}
              onChange={(val) => setContextModel(val)}
            />
            <p
              className='settings-help'
              style={{
                marginTop: '8px',
                fontSize: '11px',
                color: 'var(--text-dim)',
                lineHeight: 1.4,
              }}
            >
              The model used for background memory compression. Defaults to the
              active session model (auto). Override manually if you want a
              lighter/faster model for compression.
            </p>
          </div>

          <div
            className='settings-card'
            style={{
              marginBottom: '20px',
              padding: '12px',
              background: 'var(--bg-card)',
              borderRadius: '8px',
              border: '1px solid var(--border-color)',
            }}
          >
            <label
              className='settings-label'
              style={{
                display: 'block',
                marginBottom: '8px',
                fontSize: '13px',
                fontWeight: 600,
              }}
            >
              Semantic Embedding Model
            </label>
            <PremiumSelect
              label=''
              value={embedModel}
              options={embedModels}
              onChange={(val) => setEmbedModel(val)}
            />
            <p
              className='settings-help'
              style={{
                marginTop: '8px',
                fontSize: '11px',
                color: 'var(--text-dim)',
                lineHeight: 1.4,
              }}
            >
              The model used for semantic vector search in memory. Use a
              provider-prefixed embedding model such as `ollama:...`,
              `lmstudio:...`, `llamacpp:...`, or `openai:...` so the runtime
              uses the correct embeddings endpoint.
            </p>
          </div>

          <div
            className='settings-card'
            style={{
              marginBottom: '20px',
              padding: '12px',
              background: 'var(--bg-card)',
              borderRadius: '8px',
              border: '1px solid var(--border-color)',
            }}
          >
            <label
              className='settings-label'
              style={{
                display: 'block',
                marginBottom: '8px',
                fontSize: '13px',
                fontWeight: 600,
              }}
            >
              Granular Agentic Visibility
            </label>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: '10px',
              }}
            >
              <span style={{ fontSize: '13px', color: 'var(--text)' }}>
                Show Planner Strategy
              </span>
              <label
                className='switch'
                style={{ width: '34px', height: '20px' }}
              >
                <input
                  type='checkbox'
                  checked={showPlannerLog}
                  onChange={(e) => setShowPlannerLog(e.target.checked)}
                />
                <span className='slider round'></span>
              </label>
            </div>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: '10px',
              }}
            >
              <span style={{ fontSize: '13px', color: 'var(--text)' }}>
                Show Actor Thoughts (Reasoning)
              </span>
              <label
                className='switch'
                style={{ width: '34px', height: '20px' }}
              >
                <input
                  type='checkbox'
                  checked={showActorThought}
                  onChange={(e) => setShowActorThought(e.target.checked)}
                />
                <span className='slider round'></span>
              </label>
            </div>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <span style={{ fontSize: '13px', color: 'var(--text)' }}>
                Show Observer Activity
              </span>
              <label
                className='switch'
                style={{ width: '34px', height: '20px' }}
              >
                <input
                  type='checkbox'
                  checked={showObserverActivity}
                  onChange={(e) => setShowObserverActivity(e.target.checked)}
                />
                <span className='slider round'></span>
              </label>
            </div>
          </div>

          <div
            className='settings-card'
            style={{
              padding: '12px',
              background: 'var(--surface2)',
              borderRadius: '8px',
              border: '1px solid var(--border)',
            }}
          >
            <label
              className='settings-label'
              style={{
                display: 'block',
                marginBottom: '8px',
                fontSize: '13px',
                fontWeight: 600,
              }}
            >
              Memory Controls
            </label>
            <button
              className='memory-clear-btn'
              style={{ width: '100%', padding: '10px' }}
              onClick={() => {
                if (
                  window.confirm(
                    'Are you sure you want to purge the compressed memory of this session? Conversation history will remain.',
                  )
                ) {
                  clearSessionContext();
                }
              }}
            >
              🗑 Purge Session Context
            </button>
          </div>

          <div
            className='settings-card'
            style={{
              padding: '12px',
              background: 'var(--bg-card)',
              borderRadius: '8px',
              border: '1px solid var(--border-color)',
            }}
          >
            <label
              className='settings-label'
              style={{
                display: 'block',
                marginBottom: '8px',
                fontSize: '13px',
                fontWeight: 600,
              }}
            >
              Storage Admin
            </label>
            <p
              className='settings-help'
              style={{
                marginTop: 0,
                marginBottom: '10px',
                fontSize: '11px',
                color: 'var(--text-dim)',
                lineHeight: 1.5,
              }}
            >
              Old DBs can absolutely cause stale memory, old sessions, and
              retrieval pollution. Use this to back up and reset the exact
              stores Nova is using.
            </p>

            <div className='storage-grid'>
              {Object.entries(storageSelection).map(([key, enabled]) => {
                const info = storageStatus?.stores?.[key];
                return (
                  <label
                    key={key}
                    className={`storage-card ${enabled ? 'active' : ''}`}
                  >
                    <input
                      type='checkbox'
                      checked={enabled}
                      onChange={() =>
                        updateStoreSelection(
                          key as keyof typeof storageSelection,
                        )
                      }
                    />
                    <div>
                      <div className='storage-card-title'>
                        {key.replace(/_/g, ' ')}
                      </div>
                      <div className='storage-card-meta'>
                        {info?.exists
                          ? `${info.kind} · ${(info.size_bytes / 1024).toFixed(1)} KB`
                          : 'missing'}
                        {typeof info?.records === 'number'
                          ? ` · ${info.records} rows`
                          : ''}
                      </div>
                    </div>
                  </label>
                );
              })}
            </div>

            <input
              className='memory-search-input'
              type='text'
              placeholder='Backup label (optional)'
              value={backupLabel}
              onChange={(e) => setBackupLabel(e.target.value)}
              style={{ marginTop: '10px' }}
            />

            <label className='storage-toggle-row'>
              <input
                type='checkbox'
                checked={backupFirst}
                onChange={(e) => setBackupFirst(e.target.checked)}
              />
              <span>Backup before reset</span>
            </label>

            <label className='storage-toggle-row'>
              <input
                type='checkbox'
                checked={preserveDefaultAgent}
                onChange={(e) => setPreserveDefaultAgent(e.target.checked)}
              />
              <span>Preserve default agent profile</span>
            </label>

            <div className='memory-actions' style={{ marginTop: '10px' }}>
              <button
                onClick={loadStorage}
                className='memory-refresh-btn'
                disabled={storageLoading}
              >
                {storageLoading ? '…' : '↺ Refresh'}
              </button>
              <button
                onClick={createBackup}
                className='memory-refresh-btn'
                disabled={storageLoading}
              >
                💾 Backup
              </button>
              <button
                onClick={resetStorage}
                className='memory-clear-btn'
                disabled={storageLoading}
              >
                Reset Selected
              </button>
            </div>

            {storageMessage && (
              <div className='storage-status-note'>{storageMessage}</div>
            )}

            {storageStatus && (
              <div className='storage-path-note'>
                Backup root: <code>{storageStatus.backup_root}</code>
              </div>
            )}

            {storageBackups.length > 0 && (
              <div className='storage-backup-list'>
                {storageBackups.map((backup) => (
                  <div key={backup.path} className='memory-card'>
                    <div className='memory-triplet'>
                      <span className='memory-subject'>{backup.name}</span>
                      <span className='memory-predicate'>
                        {(backup.stores || []).join(', ') || 'no stores listed'}
                      </span>
                      <span className='memory-object'>{backup.path}</span>
                    </div>
                    <div className='memory-footer'>
                      <span className='memory-date'>
                        {backup.created_at
                          ? new Date(backup.created_at).toLocaleString()
                          : 'unknown time'}
                      </span>
                      <span className='memory-date'>
                        {(backup.size_bytes / 1024).toFixed(1)} KB
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'memory' && (
        <div className='options-section'>
          <div className='memory-card' style={{ marginBottom: '12px' }}>
            <div
              className='memory-footer'
              style={{ display: 'flex', justifyContent: 'space-between' }}
            >
              <span className='memory-date'>Memory Benchmark Harness</span>
              <button
                className='memory-refresh-btn'
                onClick={runMemoryBenchmark}
                disabled={benchmarkLoading}
              >
                {benchmarkLoading ? 'Running…' : 'Run Benchmark'}
              </button>
            </div>
            {benchmarkError && (
              <div className='memory-empty' style={{ minHeight: 'unset' }}>
                {benchmarkError}
              </div>
            )}
            {benchmarkResult ? (
              <div style={{ display: 'grid', gap: '8px', marginTop: '8px' }}>
                <div className='memory-footer'>
                  <span className='memory-date'>
                    Overall score: {(benchmarkResult.overall_score * 100).toFixed(1)}%
                  </span>
                  <span className='memory-date'>
                    {new Date(benchmarkResult.ran_at).toLocaleString()}
                  </span>
                </div>
                <div className='memory-triplet'>
                  <span className='memory-subject'>Deterministic</span>
                  <span className='memory-predicate'>
                    P {(benchmarkResult.metrics.deterministic_extraction.precision * 100).toFixed(1)}% · R{' '}
                    {(benchmarkResult.metrics.deterministic_extraction.recall * 100).toFixed(1)}%
                  </span>
                  <span className='memory-object'>
                    F1 {(benchmarkResult.metrics.deterministic_extraction.f1 * 100).toFixed(1)}% · Void{' '}
                    {(benchmarkResult.metrics.deterministic_extraction.void_accuracy * 100).toFixed(1)}%
                  </span>
                </div>
                <div className='memory-triplet'>
                  <span className='memory-subject'>Direct Fact Guard</span>
                  <span className='memory-predicate'>
                    Accuracy {(benchmarkResult.metrics.direct_fact_guard.accuracy * 100).toFixed(1)}%
                  </span>
                  <span className='memory-object'>
                    {benchmarkResult.metrics.direct_fact_guard.duration_ms.toFixed(2)} ms
                  </span>
                </div>
                <div className='memory-triplet'>
                  <span className='memory-subject'>Semantic Retrieval</span>
                  <span className='memory-predicate'>
                    Hit@3 {(benchmarkResult.metrics.semantic_retrieval.hit_rate_at_3 * 100).toFixed(1)}%
                  </span>
                  <span className='memory-object'>
                    MRR {(benchmarkResult.metrics.semantic_retrieval.mrr_at_3 * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ) : (
              <div className='memory-empty' style={{ minHeight: 'unset', marginTop: '8px' }}>
                No benchmark run yet for this owner.
              </div>
            )}
          </div>

          <div className='memory-search-row'>
            <input
              className='memory-search-input'
              type='text'
              placeholder='Semantic search…'
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && searchMemory()}
            />
            <button
              className='memory-search-btn'
              onClick={searchMemory}
              disabled={isSearching}
            >
              {isSearching ? '…' : '⌕'}
            </button>
            {searchResults !== null && (
              <button
                className='memory-clear-search'
                onClick={() => {
                  setSearchResults(null);
                  setSearchQuery('');
                }}
              >
                ✕
              </button>
            )}
          </div>

          {searchResults !== null && (
            <div className='memory-search-label'>
              {searchResults.length} semantic match
              {searchResults.length !== 1 ? 'es' : ''} for "{searchQuery}"
            </div>
          )}

          <div className='memory-list'>
            {isLoading ? (
              <div className='memory-empty'>Loading…</div>
            ) : displayMemories.length === 0 ? (
              <div className='memory-empty'>
                {searchResults !== null
                  ? 'No matches found.'
                  : 'No memories stored yet.'}
                {searchResults === null && (
                  <>
                    <br />
                    <span style={{ opacity: 0.5, fontSize: '11px' }}>
                      Ask the agent to remember something!
                    </span>
                  </>
                )}
              </div>
            ) : (
              displayMemories.map((memory) => (
                <div key={memory.id} className='memory-card'>
                  <div className='memory-triplet'>
                    <span className='memory-subject'>{memory.subject}</span>
                    <span className='memory-predicate'>{memory.predicate}</span>
                    <span className='memory-object'>{memory.object}</span>
                  </div>
                  <div className='memory-footer'>
                    <span className='memory-date'>
                      {memory.similarity !== undefined
                        ? `${Math.round(memory.similarity * 100)}% match`
                        : new Date(memory.created_at).toLocaleDateString()}
                    </span>
                    <button
                      className='memory-delete-btn'
                      onClick={() => deleteMemory(memory.id)}
                      title='Delete'
                    >
                      ✕
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>

          {total > 0 && (
            <div className='memory-actions'>
              <button onClick={loadMemories} className='memory-refresh-btn'>
                ↺ Refresh
              </button>
              <button
                onClick={clearAllMemories}
                className={`memory-clear-btn ${confirmClear ? 'confirm' : ''}`}
              >
                {confirmClear ? '⚠ Click again to confirm' : '🗑 Clear All'}
              </button>
            </div>
          )}
        </div>
      )}

      {activeTab === 'context' && (
        <div className='options-section'>
          <h4
            className='section-title'
            style={{
              fontSize: '13px',
              borderBottom: '1px solid var(--border-color)',
              paddingBottom: '4px',
              marginBottom: '10px',
            }}
          >
            Context Window
          </h4>
          {!sessionId ? (
            <div className='memory-empty'>
              No active session. Start a chat first.
            </div>
          ) : (
            <>
              {sessionMemoryState && (
                <div
                  style={{ marginBottom: '14px', display: 'grid', gap: '10px' }}
                >
                  <div className='memory-card'>
                    <div
                      className='memory-footer'
                      style={{ marginBottom: '8px' }}
                    >
                      <span className='memory-date'>
                        Trusted facts:{' '}
                        {sessionMemoryState.summary.deterministic_fact_count}
                      </span>
                      <span className='memory-date'>
                        Candidates:{' '}
                        {sessionMemoryState.summary.candidate_signal_count}
                      </span>
                    </div>
                    {sessionMemoryState.summary.stance_signal_count > 0 && (
                      <div
                        className='memory-footer'
                        style={{ marginBottom: '8px' }}
                      >
                        <span className='memory-date'>
                          Tracked stances:{' '}
                          {sessionMemoryState.summary.stance_signal_count}
                        </span>
                      </div>
                    )}
                    {sessionMemoryState.deterministic_facts.length === 0 ? (
                      <div
                        className='memory-empty'
                        style={{
                          minHeight: 'unset',
                          padding: '0',
                          textAlign: 'left',
                        }}
                      >
                        No trusted facts for this session yet.
                      </div>
                    ) : (
                      sessionMemoryState.deterministic_facts.map(
                        (fact, index) => (
                          <div
                            key={`${fact.subject}-${fact.predicate}-${fact.object}-${index}`}
                            className='memory-triplet'
                            style={{ marginBottom: '8px' }}
                          >
                            <span className='memory-subject'>
                              {fact.subject}
                            </span>
                            <span className='memory-predicate'>
                              {fact.predicate}
                            </span>
                            <span className='memory-object'>{fact.object}</span>
                          </div>
                        ),
                      )
                    )}
                  </div>

                  {sessionMemoryState.superseded_facts.length > 0 && (
                    <div className='memory-card'>
                      <div
                        className='memory-footer'
                        style={{ marginBottom: '8px' }}
                      >
                        <span className='memory-date'>Superseded facts</span>
                      </div>
                      {sessionMemoryState.superseded_facts
                        .slice(0, 4)
                        .map((fact, index) => (
                          <div
                            key={`${fact.subject}-${fact.predicate}-${fact.object}-${index}`}
                            className='memory-triplet'
                            style={{ marginBottom: '8px', opacity: 0.72 }}
                          >
                            <span className='memory-subject'>
                              {fact.subject}
                            </span>
                            <span className='memory-predicate'>
                              {fact.predicate}
                            </span>
                            <span className='memory-object'>{fact.object}</span>
                          </div>
                        ))}
                    </div>
                  )}

                  {sessionMemoryState.stance_signals.length > 0 && (
                    <div className='memory-card'>
                      <div
                        className='memory-footer'
                        style={{ marginBottom: '8px' }}
                      >
                        <span className='memory-date'>
                          Tracked user positions
                        </span>
                      </div>
                      {sessionMemoryState.stance_signals
                        .slice(0, 4)
                        .map((signal, index) => (
                          <div
                            key={`${signal.text}-${index}`}
                            className='context-line'
                            style={{ marginBottom: '8px' }}
                          >
                            <strong style={{ color: 'var(--text)' }}>
                              {signal.topic || 'stance'}:
                            </strong>{' '}
                            {signal.text}
                            <span style={{ opacity: 0.55 }}>
                              {' '}
                              · {signal.confidence || signal.reason}
                            </span>
                            {signal.superseded ? (
                              <span style={{ opacity: 0.45 }}>
                                {' '}
                                · superseded
                              </span>
                            ) : null}
                          </div>
                        ))}
                    </div>
                  )}

                  {sessionMemoryState.candidate_signals.some(
                    (signal) => signal.signal_type !== 'stance',
                  ) && (
                    <div className='memory-card'>
                      <div
                        className='memory-footer'
                        style={{ marginBottom: '8px' }}
                      >
                        <span className='memory-date'>
                          Candidate signals under review
                        </span>
                      </div>
                      {sessionMemoryState.candidate_signals
                        .filter((signal) => signal.signal_type !== 'stance')
                        .slice(0, 4)
                        .map((signal, index) => (
                          <div
                            key={`${signal.text}-${index}`}
                            className='context-line'
                            style={{ marginBottom: '6px' }}
                          >
                            {signal.text}
                            <span style={{ opacity: 0.55 }}>
                              {' '}
                              · {signal.reason}
                            </span>
                          </div>
                        ))}
                    </div>
                  )}

                  {sessionMemoryState.recent_memory_signals.length > 0 && (
                    <div className='memory-card'>
                      <div
                        className='memory-footer'
                        style={{ marginBottom: '8px' }}
                      >
                        <span className='memory-date'>
                          Why the system changed memory
                        </span>
                      </div>
                      {sessionMemoryState.recent_memory_signals
                        .slice(0, 4)
                        .map((signal, index) => (
                          <div
                            key={`${signal.event_type}-${index}`}
                            className='context-line'
                            style={{ marginBottom: '8px' }}
                          >
                            <strong style={{ color: 'var(--text)' }}>
                              {signal.label}:
                            </strong>{' '}
                            {signal.summary}
                          </div>
                        ))}
                    </div>
                  )}
                </div>
              )}

              {sessionContext.length === 0 ? (
                <div className='memory-empty'>
                  Context is empty for this session.
                </div>
              ) : (
                <>
                  <div className='context-meta'>
                    {sessionContext.length} context lines
                  </div>
                  <div className='context-list'>
                    {sessionContext.map((line, i) => (
                      <div key={i} className='context-line'>
                        {line}
                      </div>
                    ))}
                  </div>
                </>
              )}
            </>
          )}
          {sessionId && (
            <button
              className='memory-refresh-btn'
              style={{ marginTop: '8px' }}
              onClick={() => {
                loadMemoryState();
                loadContext();
              }}
            >
              ↺ Refresh
            </button>
          )}
        </div>
      )}
    </div>
  );
};
