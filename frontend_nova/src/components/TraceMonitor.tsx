import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  buildMonitorOverview as buildShovsMonitorOverview,
  buildTimelineEntries as buildShovsTimelineEntries,
  describeTraceEvent as describeShovsTraceEvent,
  humanizeTraceEvent as humanizeShovsTraceEvent,
  parsePacketSections as parseShovsPacketSections,
} from '../monitor/presentation';
import { getOwnerId } from '../owner';
import { OperatorInterventions } from './OperatorInterventions';

interface TraceEventSummary {
  id: string;
  ts: number;
  iso_ts?: string;
  agent_id: string;
  session_id: string;
  run_id?: string | null;
  event_type: string;
  pass_index?: number | null;
  size_bytes: number;
  preview?: string;
  payload_ref?: string | null;
  data?: unknown;
}

interface TraceRecentResponse {
  events: TraceEventSummary[];
  count: number;
  next_before_ts?: number | null;
}

interface TraceEventResponse {
  found: boolean;
  event: TraceEventSummary | null;
}

interface TraceStats {
  window: number;
  event_count: number;
  payload_backed_events: number;
  inline_events: number;
  total_size_bytes: number;
  avg_size_bytes: number;
  max_pass_index: number;
  event_types: Record<string, number>;
  sessions: Record<string, number>;
  latest_ts?: number | null;
}

interface RunReplaySummary {
  checkpoint_count: number;
  pass_count: number;
  artifact_count: number;
  eval_count: number;
  trace_event_count: number;
  evidence_count: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number;
}

interface RunReplayArtifact {
  artifact_id: string;
  artifact_type: string;
  label: string;
  tool_name?: string | null;
  size_bytes: number;
  preview?: string;
  created_at?: string;
}

interface RunReplayCheckpoint {
  checkpoint_id: number;
  phase: string;
  tool_turn?: number;
  status?: string;
  strategy?: string;
  notes?: string;
  tools?: string[];
  created_at?: string;
}

interface RunReplayPass {
  pass_id: number;
  phase: string;
  tool_turn?: number;
  status?: string;
  objective?: string;
  strategy?: string;
  notes?: string;
  selected_tools?: string[];
  response_preview?: string;
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  estimated_cost_usd?: number;
  cumulative_cost_usd?: number;
  created_at?: string;
}

interface RunReplayEvidence {
  source: string;
  phase: string;
  tool_turn?: number | null;
  pass_id?: number | null;
  item_id: string;
  trace_id?: string | null;
  summary: string;
  provenance?: Record<string, unknown>;
}

interface RunReplayResponse {
  found: boolean;
  run: {
    run_id: string;
    status: string;
    session_id: string;
    model: string;
    started_at?: string;
    ended_at?: string | null;
  } | null;
  summary?: RunReplaySummary;
  latest_checkpoint?: {
    phase?: string;
    status?: string;
    strategy?: string;
    notes?: string;
  } | null;
  latest_pass?: {
    phase?: string;
    status?: string;
    objective?: string;
    strategy?: string;
    notes?: string;
    selected_tools?: string[];
    response_preview?: string;
  } | null;
  checkpoints?: RunReplayCheckpoint[];
  passes?: RunReplayPass[];
  artifacts?: RunReplayArtifact[];
  evidence?: RunReplayEvidence[];
}

interface TraceMonitorProps {
  sessionId?: string | null;
  isVisible: boolean;
  isStreaming: boolean;
  pendingConfirmation?: {
    call_id: string;
    tool: string;
    arguments: Record<string, any>;
    preview: string;
    reason: string;
    created_at?: string;
  } | null;
  onApproveConfirmation: (callId: string) => void;
  onDenyConfirmation: (callId: string, reason: string) => void;
  onStopExecution: () => void;
}

interface PacketSection {
  title: string;
  body: string;
}

const TRACE_PAGE_SIZE = 120;
const VIEW_SCOPE_KEY = 'shovs_trace_scope';
const EVENT_FILTER_KEY = 'shovs_trace_filter';
const AUTO_REFRESH_KEY = 'shovs_trace_auto';
const TRACE_FOCUS_KEY = 'shovs_trace_focus';

const BASE_EVENT_TYPES = [
  'all',
  'conversation_tension',
  'stance_signals_extracted',
  'phase_context',
  'compiled_context',
  'llm_pass_start',
  'llm_prompt',
  'llm_pass_complete',
  'tool_call',
  'tool_result',
  'prompt_components',
  'assistant_response',
] as const;

function formatCurrency(value?: number): string {
  const safe = Number(value || 0);
  if (!safe) return '$0.00';
  if (safe < 0.01) return `$${safe.toFixed(4)}`;
  return `$${safe.toFixed(2)}`;
}

function formatTokens(value?: number): string {
  const safe = Number(value || 0);
  if (safe >= 1000) return `${(safe / 1000).toFixed(1)}k`;
  return String(safe);
}

function formatTime(ts?: number): string {
  if (!ts) return '--';
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function isPromptMessage(
  item: unknown,
): item is { role?: string; content?: string } {
  return typeof item === 'object' && item !== null;
}

function mergeEvents(
  existing: TraceEventSummary[],
  incoming: TraceEventSummary[],
): TraceEventSummary[] {
  const byId = new Map<string, TraceEventSummary>();
  for (const event of existing) byId.set(event.id, event);
  for (const event of incoming) byId.set(event.id, event);

  const merged = Array.from(byId.values());
  merged.sort((a, b) => b.ts - a.ts);
  return merged;
}

export const TraceMonitor: React.FC<TraceMonitorProps> = ({
  sessionId,
  isVisible,
  isStreaming,
  pendingConfirmation,
  onApproveConfirmation,
  onDenyConfirmation,
  onStopExecution,
}) => {
  const [focusMode, setFocusMode] = useState<'story' | 'passes' | 'inspect'>(
    () => {
      const stored = localStorage.getItem(TRACE_FOCUS_KEY);
      if (stored === 'inspect' || stored === 'passes') return stored;
      return 'story';
    },
  );
  const [scope, setScope] = useState<'session' | 'all'>(() => {
    const stored = localStorage.getItem(VIEW_SCOPE_KEY);
    return stored === 'all' ? 'all' : 'session';
  });
  const [eventType, setEventType] = useState<string>(() => {
    const stored = localStorage.getItem(EVENT_FILTER_KEY) || 'all';
    return stored === 'story' ? 'all' : stored;
  });
  const [autoRefresh, setAutoRefresh] = useState<boolean>(
    () => localStorage.getItem(AUTO_REFRESH_KEY) !== 'false',
  );

  const [events, setEvents] = useState<TraceEventSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<TraceEventSummary | null>(
    null,
  );
  const [stats, setStats] = useState<TraceStats | null>(null);
  const [runReplay, setRunReplay] = useState<RunReplayResponse | null>(null);

  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [loadingRunReplay, setLoadingRunReplay] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const scopedSessionId =
    scope === 'session' ? sessionId || undefined : undefined;

  const eventTypeOptions = useMemo(() => {
    const runtimeTypes = stats ? Object.keys(stats.event_types).sort() : [];
    const merged = new Set<string>(BASE_EVENT_TYPES);
    for (const type of runtimeTypes) merged.add(type);
    return Array.from(merged);
  }, [stats]);

  const filteredEvents = useMemo(() => {
    const baseEvents =
      eventType === 'all'
        ? events
        : events.filter((event) => event.event_type === eventType);
    if (!search.trim()) return baseEvents;
    const needle = search.trim().toLowerCase();
    return baseEvents.filter((event) => {
      const haystack = [
        event.event_type,
        event.session_id,
        event.agent_id,
        event.preview || '',
        describeShovsTraceEvent(event),
      ]
        .join(' ')
        .toLowerCase();
      return haystack.includes(needle);
    });
  }, [eventType, events, search]);

  const oldestTs = events.length ? events[events.length - 1].ts : undefined;
  const overview = useMemo(
    () =>
      buildShovsMonitorOverview({
        events,
        runReplay,
        pendingConfirmation,
      }),
    [events, pendingConfirmation, runReplay],
  );
  const recentTimelineEntries = useMemo(
    () => buildShovsTimelineEntries(events, 6),
    [events],
  );
  const recentReplayPasses = useMemo(
    () => [...(runReplay?.passes || [])].slice(-8).reverse(),
    [runReplay],
  );
  const runStoryCards = useMemo(() => {
    const items: Array<{
      id: string;
      title: string;
      eyebrow: string;
      summary: string;
      detail?: string;
    }> = [];
    if (runReplay?.latest_pass) {
      items.push({
        id: 'latest-pass',
        title: runReplay.latest_pass.phase || 'Latest pass',
        eyebrow: 'Latest pass',
        summary:
          runReplay.latest_pass.objective ||
          runReplay.latest_pass.strategy ||
          runReplay.latest_pass.response_preview ||
          'No pass summary stored.',
        detail:
          (runReplay.latest_pass.selected_tools || []).length > 0
            ? `tools: ${(runReplay.latest_pass.selected_tools || []).join(', ')}`
            : undefined,
      });
    }
    if ((runReplay?.evidence || []).length > 0) {
      const firstEvidence = runReplay?.evidence?.[0];
      items.push({
        id: 'latest-evidence',
        title: firstEvidence?.phase || 'Evidence',
        eyebrow: 'Best evidence',
        summary: firstEvidence?.summary || 'Evidence was collected.',
        detail: firstEvidence?.source || undefined,
      });
    }
    for (const entry of recentTimelineEntries.slice(0, 4)) {
      items.push({
        id: `timeline-${entry.id}`,
        title: entry.stage,
        eyebrow: entry.passLabel || 'Timeline',
        summary: entry.headline,
        detail: entry.lines.join(' · '),
      });
    }
    return items.slice(0, 6);
  }, [recentTimelineEntries, runReplay]);

  const fetchRecent = useCallback(
    async (opts?: { append?: boolean; beforeTs?: number }) => {
      const params = new URLSearchParams();
      params.set('limit', String(TRACE_PAGE_SIZE));
      params.set('owner_id', getOwnerId());
      if (scopedSessionId) params.set('session_id', scopedSessionId);
      if (eventType && eventType !== 'all') params.set('event_type', eventType);
      if (opts?.beforeTs) params.set('before_ts', String(opts.beforeTs));

      const res = await fetch(`/api/logs/traces/recent?${params.toString()}`);
      if (!res.ok) {
        throw new Error(`Trace list failed: HTTP ${res.status}`);
      }

      const data: TraceRecentResponse = await res.json();
      const incoming = Array.isArray(data.events) ? data.events : [];

      setEvents((prev) => {
        if (opts?.append) {
          return mergeEvents(prev, incoming);
        }
        return incoming;
      });
    },
    [eventType, scopedSessionId],
  );

  const fetchStats = useCallback(async () => {
    const params = new URLSearchParams();
    params.set('window', '500');
    params.set('owner_id', getOwnerId());
    if (scopedSessionId) params.set('session_id', scopedSessionId);

    const res = await fetch(`/api/logs/traces/stats?${params.toString()}`);
    if (!res.ok) {
      throw new Error(`Trace stats failed: HTTP ${res.status}`);
    }
    const data: TraceStats = await res.json();
    setStats(data);
  }, [scopedSessionId]);

  const refreshAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await Promise.all([fetchRecent(), fetchStats()]);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to load trace monitor data',
      );
    } finally {
      setLoading(false);
    }
  }, [fetchRecent, fetchStats]);

  const loadOlder = useCallback(async () => {
    if (!oldestTs) return;
    setLoadingOlder(true);
    setError(null);
    try {
      await fetchRecent({ append: true, beforeTs: oldestTs });
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to load older trace events',
      );
    } finally {
      setLoadingOlder(false);
    }
  }, [fetchRecent, oldestTs]);

  useEffect(() => {
    localStorage.setItem(VIEW_SCOPE_KEY, scope);
  }, [scope]);

  useEffect(() => {
    localStorage.setItem(EVENT_FILTER_KEY, eventType);
  }, [eventType]);

  useEffect(() => {
    localStorage.setItem(AUTO_REFRESH_KEY, autoRefresh ? 'true' : 'false');
  }, [autoRefresh]);

  useEffect(() => {
    localStorage.setItem(TRACE_FOCUS_KEY, focusMode);
  }, [focusMode]);

  useEffect(() => {
    if (!isVisible) return;
    refreshAll();
  }, [isVisible, refreshAll]);

  useEffect(() => {
    if (!isVisible || !autoRefresh) return;
    const id = window.setInterval(() => {
      fetchRecent().catch(() => undefined);
      fetchStats().catch(() => undefined);
    }, 2500);
    return () => window.clearInterval(id);
  }, [autoRefresh, fetchRecent, fetchStats, isVisible]);

  useEffect(() => {
    if (!events.length) {
      setSelectedId(null);
      return;
    }

    if (!selectedId || !events.some((event) => event.id === selectedId)) {
      setSelectedId(events[0].id);
    }
  }, [events, selectedId]);

  useEffect(() => {
    if (!selectedId || !isVisible) {
      setSelectedEvent(null);
      return;
    }

    const controller = new AbortController();
    setLoadingDetail(true);

    fetch(
      `/api/logs/traces/event/${encodeURIComponent(selectedId)}?owner_id=${encodeURIComponent(getOwnerId())}`,
      {
        signal: controller.signal,
      },
    )
      .then(async (res) => {
        if (!res.ok) throw new Error(`Trace detail failed: HTTP ${res.status}`);
        const data: TraceEventResponse = await res.json();
        setSelectedEvent(data.found ? data.event : null);
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setError(
          err instanceof Error ? err.message : 'Failed to load event detail',
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingDetail(false);
      });

    return () => controller.abort();
  }, [isVisible, selectedId]);

  useEffect(() => {
    if (!isVisible || !selectedEvent?.run_id) {
      setRunReplay(null);
      setLoadingRunReplay(false);
      return;
    }

    const controller = new AbortController();
    setLoadingRunReplay(true);

    fetch(
      `/api/logs/traces/run/${encodeURIComponent(selectedEvent.run_id)}?owner_id=${encodeURIComponent(getOwnerId())}&trace_limit=180`,
      { signal: controller.signal },
    )
      .then(async (res) => {
        if (!res.ok) throw new Error(`Run replay failed: HTTP ${res.status}`);
        const data: RunReplayResponse = await res.json();
        setRunReplay(data.found ? data : null);
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setRunReplay(null);
        setError(
          err instanceof Error ? err.message : 'Failed to load run replay',
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingRunReplay(false);
      });

    return () => controller.abort();
  }, [isVisible, selectedEvent?.run_id]);

  const promptMessages = useMemo(() => {
    if (!selectedEvent || selectedEvent.event_type !== 'llm_prompt')
      return [] as Array<{ role?: string; content?: string }>;
    const data = selectedEvent.data;
    if (!data || typeof data !== 'object')
      return [] as Array<{ role?: string; content?: string }>;

    const maybeMessages = (data as { messages?: unknown }).messages;
    if (!Array.isArray(maybeMessages))
      return [] as Array<{ role?: string; content?: string }>;

    return maybeMessages.filter(isPromptMessage);
  }, [selectedEvent]);

  const eventJson = useMemo(() => {
    if (!selectedEvent) return '';
    return JSON.stringify(selectedEvent.data ?? {}, null, 2);
  }, [selectedEvent]);
  const selectedSummary = useMemo(
    () => (selectedEvent ? describeShovsTraceEvent(selectedEvent) : ''),
    [selectedEvent],
  );
  const selectedPacketSections = useMemo(() => {
    if (!selectedEvent) return [] as PacketSection[];
    if (
      selectedEvent.event_type !== 'phase_context' &&
      selectedEvent.event_type !== 'compiled_context'
    ) {
      return [] as PacketSection[];
    }
    const data =
      selectedEvent.data && typeof selectedEvent.data === 'object'
        ? (selectedEvent.data as Record<string, unknown>)
        : null;
    return parseShovsPacketSections(String(data?.content || ''));
  }, [selectedEvent]);
  const selectedIncludedItems = useMemo(() => {
    if (!selectedEvent?.data || typeof selectedEvent.data !== 'object')
      return [];
    const included = (selectedEvent.data as Record<string, unknown>).included;
    if (!Array.isArray(included)) return [];
    return included.filter(
      (item): item is Record<string, any> =>
        typeof item === 'object' && item !== null,
    );
  }, [selectedEvent]);
  const recentReplayCheckpoints = useMemo(
    () => [...(runReplay?.checkpoints || [])].slice(-4).reverse(),
    [runReplay],
  );

  return (
    <div className='trace-monitor-shell'>
      <div className='trace-monitor-head'>
        <div className='trace-title-wrap'>
          <span className='trace-title'>Trace Monitor</span>
          <span className='trace-subtitle'>
            Human-readable run timeline with optional deep payload inspection
          </span>
        </div>

        <div className='trace-controls'>
          <div className='trace-mode-switch'>
            <button
              className={focusMode === 'story' ? 'active' : ''}
              onClick={() => setFocusMode('story')}
            >
              Story
            </button>
            <button
              className={focusMode === 'passes' ? 'active' : ''}
              onClick={() => setFocusMode('passes')}
            >
              Passes
            </button>
            <button
              className={focusMode === 'inspect' ? 'active' : ''}
              onClick={() => setFocusMode('inspect')}
            >
              Inspect
            </button>
          </div>
          <label className='trace-control'>
            <span>scope</span>
            <select
              value={scope}
              onChange={(e) =>
                setScope(e.target.value === 'all' ? 'all' : 'session')
              }
            >
              <option value='session'>current session</option>
              <option value='all'>all sessions</option>
            </select>
          </label>

          <label className='trace-control'>
            <span>event</span>
            <select
              value={eventType}
              onChange={(e) => setEventType(e.target.value)}
            >
              {eventTypeOptions.map((type) => (
                <option key={type} value={type}>
                  {humanizeShovsTraceEvent(type)}
                </option>
              ))}
            </select>
          </label>

          <label className='trace-toggle'>
            <input
              type='checkbox'
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            <span>live</span>
          </label>

          <button
            className='trace-action'
            onClick={refreshAll}
            disabled={loading}
          >
            {loading ? 'refreshing...' : 'refresh'}
          </button>
        </div>
      </div>

      <OperatorInterventions
        sessionId={sessionId}
        isStreaming={isStreaming}
        pendingConfirmation={pendingConfirmation}
        onApprove={onApproveConfirmation}
        onDeny={onDenyConfirmation}
        onStop={onStopExecution}
        variant='compact'
      />

      <div className='trace-stats-row'>
        <div className='trace-stat-card'>
          <span className='k'>events</span>
          <span className='v'>{stats?.event_count ?? 0}</span>
        </div>
        <div className='trace-stat-card'>
          <span className='k'>payload-backed</span>
          <span className='v'>{stats?.payload_backed_events ?? 0}</span>
        </div>
        <div className='trace-stat-card'>
          <span className='k'>avg size</span>
          <span className='v'>{formatBytes(stats?.avg_size_bytes ?? 0)}</span>
        </div>
        <div className='trace-stat-card'>
          <span className='k'>max pass</span>
          <span className='v'>
            {(stats?.max_pass_index ?? -1) >= 0 ? stats?.max_pass_index : '--'}
          </span>
        </div>
        <div className='trace-stat-card'>
          <span className='k'>run tokens</span>
          <span className='v'>
            {formatTokens(runReplay?.summary?.total_tokens ?? 0)}
          </span>
        </div>
        <div className='trace-stat-card'>
          <span className='k'>run cost</span>
          <span className='v'>
            {formatCurrency(runReplay?.summary?.estimated_cost_usd ?? 0)}
          </span>
        </div>
      </div>

      {error && <div className='trace-error'>{error}</div>}
      {focusMode === 'story' ? (
        <div className='trace-overview-shell'>
          <section className='trace-overview-section'>
            <div className='trace-section-head'>
              <div className='trace-section-title'>Overview</div>
              <div className='trace-section-subtitle'>
                Current run state in operator language
              </div>
            </div>
            <div className='trace-overview-grid primary'>
              {overview.primary.map((card) => (
                <div
                  key={card.id}
                  className={`trace-overview-card tone-${card.tone}`}
                >
                  <div className='trace-overview-eyebrow'>
                    {card.eyebrow || card.title}
                  </div>
                  <div className='trace-overview-title'>{card.title}</div>
                  <div className='trace-overview-summary'>{card.summary}</div>
                  {card.detail ? (
                    <div className='trace-overview-detail'>{card.detail}</div>
                  ) : null}
                </div>
              ))}
            </div>
            {overview.secondary.length > 0 ? (
              <div className='trace-overview-grid secondary'>
                {overview.secondary.map((card) => (
                  <div
                    key={card.id}
                    className={`trace-overview-card tone-${card.tone}`}
                  >
                    <div className='trace-overview-eyebrow'>
                      {card.eyebrow || card.title}
                    </div>
                    <div className='trace-overview-title'>{card.title}</div>
                    <div className='trace-overview-summary'>{card.summary}</div>
                  </div>
                ))}
              </div>
            ) : null}
          </section>

          <section className='trace-overview-section'>
            <div className='trace-section-head'>
              <div className='trace-section-title'>Run Story</div>
              <div className='trace-section-subtitle'>
                One condensed path through what happened, when, and why
              </div>
            </div>
            <div className='trace-replay-grid'>
              {runStoryCards.length === 0 ? (
                <div className='trace-empty'>No run story available yet.</div>
              ) : (
                runStoryCards.map((card) => (
                  <div key={card.id} className='trace-replay-panel'>
                    <div className='trace-replay-panel-title'>
                      {card.eyebrow}
                    </div>
                    <div className='trace-replay-panel-list'>
                      <div className='trace-replay-card tone-neutral'>
                        <div className='trace-replay-card-top'>
                          <span>{card.title}</span>
                        </div>
                        <div className='trace-replay-card-summary'>
                          {card.summary}
                        </div>
                        {card.detail ? (
                          <div className='trace-replay-card-detail'>
                            {card.detail}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      ) : focusMode === 'passes' ? (
        <div className='trace-overview-shell'>
          <section className='trace-overview-section'>
            <div className='trace-list-head timeline-head'>
              <div className='trace-section-title'>Passes and Cost</div>
              <span className='trace-count'>{recentReplayPasses.length}</span>
            </div>
            <div className='trace-run-list'>
              {recentReplayPasses.length === 0 ? (
                <div className='trace-empty'>No run passes stored yet.</div>
              ) : (
                recentReplayPasses.map((pass) => (
                  <div key={`pass-${pass.pass_id}`} className='trace-run-card'>
                    <div className='trace-run-card-head'>
                      <span>
                        {pass.phase || 'phase'} · {pass.status || '--'}
                      </span>
                      <span>
                        {formatTime(
                          pass.created_at
                            ? Date.parse(pass.created_at) / 1000
                            : undefined,
                        )}
                      </span>
                    </div>
                    <div className='trace-run-card-copy'>
                      {(
                        pass.objective ||
                        pass.strategy ||
                        pass.response_preview ||
                        'No pass summary stored.'
                      ).trim()}
                    </div>
                    <div className='trace-timeline-meta'>
                      <span>in {formatTokens(pass.input_tokens || 0)}</span>
                      <span>out {formatTokens(pass.output_tokens || 0)}</span>
                      <span>total {formatTokens(pass.total_tokens || 0)}</span>
                      <span>
                        turn {formatCurrency(pass.estimated_cost_usd || 0)}
                      </span>
                      <span>
                        run {formatCurrency(pass.cumulative_cost_usd || 0)}
                      </span>
                    </div>
                    {(pass.selected_tools || []).length > 0 ? (
                      <div className='trace-run-card-copy'>
                        tools: {(pass.selected_tools || []).join(', ')}
                      </div>
                    ) : null}
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      ) : (
        <div className='trace-grid inspect-focus'>
          <section className='trace-list-pane'>
            <div className='trace-list-head'>
              <input
                className='trace-search'
                placeholder='search events'
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
              <span className='trace-count'>{filteredEvents.length}</span>
            </div>

            <div className='trace-list'>
              {filteredEvents.length === 0 ? (
                <div className='trace-empty'>
                  No trace events for this filter.
                </div>
              ) : (
                filteredEvents.map((event) => (
                  <button
                    key={event.id}
                    className={`trace-row ${selectedId === event.id ? 'active' : ''}`}
                    onClick={() => setSelectedId(event.id)}
                  >
                    <div className='trace-row-top'>
                      <span className='t-type'>
                        {humanizeShovsTraceEvent(event.event_type)}
                      </span>
                      <span className='t-time'>{formatTime(event.ts)}</span>
                    </div>
                    <div className='trace-row-meta'>
                      <span>
                        pass{' '}
                        {typeof event.pass_index === 'number'
                          ? event.pass_index
                          : '--'}
                      </span>
                      <span>{formatBytes(event.size_bytes || 0)}</span>
                      <span>{event.payload_ref ? 'blob' : 'inline'}</span>
                    </div>
                    <div className='trace-row-preview'>
                      {describeShovsTraceEvent(event)}
                    </div>
                  </button>
                ))
              )}
            </div>

            <div className='trace-list-foot'>
              <button
                className='trace-action'
                onClick={loadOlder}
                disabled={!oldestTs || loadingOlder}
              >
                {loadingOlder ? 'loading...' : 'load older'}
              </button>
            </div>
          </section>

          <section className='trace-detail-pane'>
            {!selectedEvent && !loadingDetail && (
              <div className='trace-empty'>
                Select an event to inspect details.
              </div>
            )}

            {loadingDetail && (
              <div className='trace-empty'>Loading event...</div>
            )}

            {selectedEvent && (
              <>
                <div className='trace-detail-head'>
                  <div className='trace-detail-title'>
                    {humanizeShovsTraceEvent(selectedEvent.event_type)}
                  </div>
                  <div className='trace-detail-meta'>
                    <span>session: {selectedEvent.session_id}</span>
                    <span>run: {selectedEvent.run_id || '--'}</span>
                    <span>agent: {selectedEvent.agent_id}</span>
                    <span>
                      pass:{' '}
                      {typeof selectedEvent.pass_index === 'number'
                        ? selectedEvent.pass_index
                        : '--'}
                    </span>
                    <span>time: {formatTime(selectedEvent.ts)}</span>
                  </div>
                </div>

                <div className='trace-json-wrap'>
                  <div className='trace-json-head'>Readable Summary</div>
                  <pre className='trace-json'>{selectedSummary}</pre>
                </div>

                {selectedPacketSections.length > 0 && (
                  <div className='trace-packet-wrap'>
                    <div className='trace-json-head'>Compiled Packet</div>
                    {selectedIncludedItems.length > 0 && (
                      <div className='trace-packet-badges'>
                        {selectedIncludedItems.map((item) => (
                          <span
                            key={`${String(item.item_id || item.title || 'item')}-${String(item.trace_id || '')}`}
                            className='trace-packet-badge'
                          >
                            {String(item.kind || 'item')} ·{' '}
                            {String(item.title || item.item_id || 'section')}
                          </span>
                        ))}
                      </div>
                    )}
                    <div className='trace-packet-list'>
                      {selectedPacketSections.map((section, index) => (
                        <div
                          key={`${section.title}-${index}`}
                          className='trace-packet-card'
                        >
                          <div className='trace-packet-title'>
                            {section.title}
                          </div>
                          <pre className='trace-packet-copy'>
                            {section.body}
                          </pre>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {(loadingRunReplay || runReplay) && (
                  <div className='trace-run-wrap'>
                    <div className='trace-json-head'>Run Replay</div>
                    {loadingRunReplay && !runReplay ? (
                      <div className='trace-empty'>Loading run replay...</div>
                    ) : (
                      <>
                        <div className='trace-run-head'>
                          <div className='trace-run-stats'>
                            <span className='trace-run-stat'>
                              status: {runReplay?.run?.status || '--'}
                            </span>
                            <span className='trace-run-stat'>
                              checkpoints:{' '}
                              {runReplay?.summary?.checkpoint_count ?? 0}
                            </span>
                            <span className='trace-run-stat'>
                              passes: {runReplay?.summary?.pass_count ?? 0}
                            </span>
                            <span className='trace-run-stat'>
                              artifacts:{' '}
                              {runReplay?.summary?.artifact_count ?? 0}
                            </span>
                            <span className='trace-run-stat'>
                              evidence:{' '}
                              {runReplay?.summary?.evidence_count ?? 0}
                            </span>
                          </div>
                          {runReplay?.latest_pass?.objective && (
                            <div className='trace-run-copy'>
                              objective: {runReplay.latest_pass.objective}
                            </div>
                          )}
                          {runReplay?.latest_checkpoint?.strategy && (
                            <div className='trace-run-copy'>
                              strategy: {runReplay.latest_checkpoint.strategy}
                            </div>
                          )}
                        </div>

                        {(runReplay?.latest_checkpoint?.notes ||
                          (runReplay?.latest_pass?.selected_tools || [])
                            .length > 0) && (
                          <div className='trace-run-head'>
                            {runReplay?.latest_checkpoint?.notes && (
                              <div className='trace-run-copy'>
                                notes: {runReplay.latest_checkpoint.notes}
                              </div>
                            )}
                            {(runReplay?.latest_pass?.selected_tools || [])
                              .length > 0 && (
                              <div className='trace-run-copy'>
                                selected tools:{' '}
                                {runReplay?.latest_pass?.selected_tools?.join(
                                  ', ',
                                )}
                              </div>
                            )}
                          </div>
                        )}

                        {recentReplayCheckpoints.length > 0 && (
                          <div className='trace-run-list'>
                            {recentReplayCheckpoints.map((checkpoint) => (
                              <div
                                key={`checkpoint-${checkpoint.checkpoint_id}`}
                                className='trace-run-card'
                              >
                                <div className='trace-run-card-head'>
                                  <span>
                                    {checkpoint.phase || 'phase'} · status{' '}
                                    {checkpoint.status || '--'}
                                  </span>
                                  <span>
                                    turn{' '}
                                    {typeof checkpoint.tool_turn === 'number'
                                      ? checkpoint.tool_turn
                                      : '--'}
                                  </span>
                                </div>
                                <div className='trace-run-card-copy'>
                                  {(
                                    checkpoint.strategy ||
                                    checkpoint.notes ||
                                    'No checkpoint notes stored.'
                                  ).trim()}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}

                        {recentReplayPasses.length > 0 && (
                          <div className='trace-run-list'>
                            {recentReplayPasses.map((pass) => (
                              <div
                                key={`pass-${pass.pass_id}`}
                                className='trace-run-card'
                              >
                                <div className='trace-run-card-head'>
                                  <span>
                                    {pass.phase || 'phase'} ·{' '}
                                    {pass.status || '--'}
                                  </span>
                                  <span>pass {pass.pass_id}</span>
                                </div>
                                <div className='trace-run-card-copy'>
                                  {(
                                    pass.objective ||
                                    pass.strategy ||
                                    pass.response_preview ||
                                    'No pass summary stored.'
                                  ).trim()}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}

                        {(runReplay?.evidence?.length || 0) > 0 && (
                          <div className='trace-run-list'>
                            {runReplay!
                              .evidence!.slice(0, 3)
                              .map((item, index) => (
                                <div
                                  key={`${item.trace_id || item.item_id}-${index}`}
                                  className='trace-run-card'
                                >
                                  <div className='trace-run-card-head'>
                                    <span>{item.phase || 'phase'}</span>
                                    <span>
                                      {item.source}
                                      {typeof item.tool_turn === 'number'
                                        ? ` · turn ${item.tool_turn}`
                                        : ''}
                                    </span>
                                  </div>
                                  <div className='trace-run-card-copy'>
                                    {item.summary || 'No evidence summary.'}
                                  </div>
                                </div>
                              ))}
                          </div>
                        )}

                        {(runReplay?.artifacts?.length || 0) > 0 && (
                          <div className='trace-run-list'>
                            {runReplay!
                              .artifacts!.slice(0, 4)
                              .map((artifact) => (
                                <div
                                  key={artifact.artifact_id}
                                  className='trace-run-card'
                                >
                                  <div className='trace-run-card-head'>
                                    <span>{artifact.label}</span>
                                    <span>
                                      {artifact.artifact_type}
                                      {artifact.tool_name
                                        ? ` · ${artifact.tool_name}`
                                        : ''}
                                    </span>
                                  </div>
                                  <div className='trace-run-card-copy'>
                                    {(artifact.preview || '').trim() ||
                                      'No preview stored.'}
                                  </div>
                                </div>
                              ))}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}

                {promptMessages.length > 0 && (
                  <div className='prompt-stack'>
                    {promptMessages.map((msg, index) => {
                      const role = String(msg.role || 'unknown');
                      const content = String(msg.content || '');
                      return (
                        <details
                          key={`${role}-${index}`}
                          className='prompt-item'
                        >
                          <summary>
                            <span className={`prompt-role role-${role}`}>
                              {role}
                            </span>
                            <span className='prompt-size'>
                              {formatBytes(content.length)}
                            </span>
                          </summary>
                          <pre>{content}</pre>
                        </details>
                      );
                    })}
                  </div>
                )}

                <div className='trace-json-wrap'>
                  <div className='trace-json-head'>Raw event payload</div>
                  <pre className='trace-json'>{eventJson}</pre>
                </div>
              </>
            )}
          </section>
        </div>
      )}
    </div>
  );
};
