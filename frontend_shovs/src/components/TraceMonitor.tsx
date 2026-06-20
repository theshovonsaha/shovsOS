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

interface RunReplayEval {
  eval_id: string;
  eval_type: string;
  phase: string;
  passed: boolean;
  score?: number | null;
  detail?: string;
  metadata?: Record<string, unknown> | null;
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

interface OperatorStoryLane {
  id: string;
  label: string;
  status: 'idle' | 'done' | 'attention' | string;
  count: number;
  event_id?: string | null;
  event_type?: string | null;
  phase?: string;
  summary: string;
}

interface OperatorStory {
  status: string;
  objective: string;
  next_best_action?: string;
  artifact_count?: number;
  cost?: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    estimated_cost_usd: number;
  };
  lanes: OperatorStoryLane[];
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
  evals?: RunReplayEval[];
  evidence?: RunReplayEvidence[];
  operator_story?: OperatorStory;
}

interface TraceMonitorProps {
  sessionId?: string | null;
  isVisible: boolean;
  isStreaming: boolean;
  pendingConfirmation?: {
    call_id: string;
    tool: string;
    arguments: Record<string, unknown>;
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

type TraceSeverity =
  | 'all'
  | 'info'
  | 'success'
  | 'warning'
  | 'error'
  | 'retrying'
  | 'blocked';

const TRACE_PAGE_SIZE = 120;
const VIEW_SCOPE_KEY = 'shovs_trace_scope';
const EVENT_FILTER_KEY = 'shovs_trace_filter';
const AUTO_REFRESH_KEY = 'shovs_trace_auto';
const TRACE_FOCUS_KEY = 'shovs_trace_focus';

const BASE_EVENT_TYPES = [
  'all',
  'conversation_tension',
  'stance_signals_extracted',
  'run_ledger',
  'phase_packet',
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

function tracePhase(event: TraceEventSummary): string {
  const data =
    event.data && typeof event.data === 'object'
      ? (event.data as Record<string, unknown>)
      : {};
  const phase = String(data.phase || data.trace_phase || '').trim();
  if (phase) return phase;
  switch (event.event_type) {
    case 'route_decision':
    case 'plan':
      return 'planning';
    case 'phase_context':
    case 'phase_packet':
    case 'compiled_context':
    case 'prompt_components':
    case 'llm_prompt':
      return 'context';
    case 'llm_pass_start':
    case 'llm_pass_complete':
      return 'model';
    case 'tool_call':
    case 'tool_result':
      return 'tool';
    case 'run_ledger':
      return 'ledger';
    case 'verification_result':
    case 'verification_warning':
      return 'verification';
    case 'assistant_response':
      return 'response';
    case 'memory_write_policy':
    case 'memory_commit_plan':
      return 'memory';
    default:
      return 'not recorded';
  }
}

function traceSeverity(event: TraceEventSummary): Exclude<TraceSeverity, 'all'> {
  const data =
    event.data && typeof event.data === 'object'
      ? (event.data as Record<string, unknown>)
      : {};
  const status = String(data.status || data.outcome || '').toLowerCase();
  const eventType = event.event_type.toLowerCase();
  const preview = String(event.preview || '').toLowerCase();
  if (
    eventType.includes('warning') ||
    status.includes('warn') ||
    preview.includes('warning')
  )
    return 'warning';
  if (
    eventType.includes('error') ||
    status.includes('error') ||
    status.includes('fail') ||
    data.success === false
  )
    return 'error';
  if (status.includes('blocked') || eventType.includes('blocked'))
    return 'blocked';
  if (status.includes('retry') || eventType.includes('redraft'))
    return 'retrying';
  if (
    status.includes('complete') ||
    status.includes('ok') ||
    status.includes('success') ||
    eventType === 'tool_result' ||
    eventType === 'assistant_response' ||
    eventType === 'verification_result'
  )
    return 'success';
  return 'info';
}

function relatedTraceIds(event: TraceEventSummary): string[] {
  const data =
    event.data && typeof event.data === 'object'
      ? (event.data as Record<string, unknown>)
      : {};
  return [
    event.run_id ? `run ${event.run_id.slice(0, 10)}` : '',
    event.session_id ? `session ${event.session_id.slice(0, 8)}` : '',
    String(data.tool_name || data.tool || ''),
    String(data.model || ''),
  ].filter(Boolean) as string[];
}

function extractRunLedger(data: unknown): Record<string, unknown> | null {
  if (!data || typeof data !== 'object') return null;
  const record = data as Record<string, unknown>;
  const nested = record.run_ledger;
  if (nested && typeof nested === 'object') return nested as Record<string, unknown>;
  if (record.version === 'run-ledger-v1' || record.ledger_mode) return record;
  return null;
}

function ledgerList(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value)
    ? value.filter(
        (item): item is Record<string, unknown> =>
          typeof item === 'object' && item !== null,
      )
    : [];
}

function eventStatusClass(status?: string): string {
  const normalized = String(status || 'idle').toLowerCase();
  if (normalized.includes('attention') || normalized.includes('warn')) return 'attention';
  if (normalized.includes('done') || normalized.includes('success') || normalized.includes('complete')) return 'done';
  return 'idle';
}

async function copyText(value: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(value);
  } catch {
    // Clipboard can be unavailable in sandboxed or insecure contexts.
  }
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
  const [phaseFilter, setPhaseFilter] = useState<string>('all');
  const [severityFilter, setSeverityFilter] =
    useState<TraceSeverity>('all');
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
    const phaseEvents =
      phaseFilter === 'all'
        ? baseEvents
        : baseEvents.filter((event) => tracePhase(event) === phaseFilter);
    const severityEvents =
      severityFilter === 'all'
        ? phaseEvents
        : phaseEvents.filter(
            (event) => traceSeverity(event) === severityFilter,
          );
    if (!search.trim()) return severityEvents;
    const needle = search.trim().toLowerCase();
    return severityEvents.filter((event) => {
      const haystack = [
        event.event_type,
        event.session_id,
        event.agent_id,
        event.run_id || '',
        tracePhase(event),
        traceSeverity(event),
        event.preview || '',
        describeShovsTraceEvent(event),
        ...relatedTraceIds(event),
      ]
        .join(' ')
        .toLowerCase();
      return haystack.includes(needle);
    });
  }, [eventType, events, phaseFilter, search, severityFilter]);

  const phaseOptions = useMemo(() => {
    const phases = new Set(events.map(tracePhase));
    return ['all', ...Array.from(phases).sort()];
  }, [events]);

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

  const operatorStory = runReplay?.operator_story;
  const fallbackLanes = useMemo<OperatorStoryLane[]>(() => {
    const stageMap: Array<{ id: string; label: string; types: string[] }> = [
      { id: 'plan', label: 'Plan', types: ['plan', 'plan_steps', 'continuation_gate'] },
      { id: 'context', label: 'Context', types: ['phase_packet', 'phase_context', 'compiled_context'] },
      { id: 'tool', label: 'Tools', types: ['tool_call', 'tool_result'] },
      { id: 'verify', label: 'Verify', types: ['verification_result', 'verification_warning'] },
      { id: 'response', label: 'Response', types: ['assistant_response'] },
    ];
    return stageMap.map((stage) => {
      const stageEvents = events.filter((event) => stage.types.includes(event.event_type));
      const latest = stageEvents[0];
      return {
        id: stage.id,
        label: stage.label,
        status: latest ? (traceSeverity(latest) === 'error' || traceSeverity(latest) === 'warning' ? 'attention' : 'done') : 'idle',
        count: stageEvents.length,
        event_id: latest?.id,
        event_type: latest?.event_type,
        phase: latest ? tracePhase(latest) : '',
        summary: latest ? describeShovsTraceEvent(latest) : 'Not recorded yet.',
      };
    });
  }, [events]);
  const storyLanes = operatorStory?.lanes?.length ? operatorStory.lanes : fallbackLanes;

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
  const selectedRunLedger = useMemo(
    () => extractRunLedger(selectedEvent?.data),
    [selectedEvent],
  );
  const selectedPacketSections = useMemo(() => {
    if (!selectedEvent) return [] as PacketSection[];
    if (
      selectedEvent.event_type !== 'phase_context' &&
      selectedEvent.event_type !== 'phase_packet' &&
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
      (item): item is Record<string, unknown> =>
        typeof item === 'object' && item !== null,
    );
  }, [selectedEvent]);
  const recentReplayCheckpoints = useMemo(
    () => [...(runReplay?.checkpoints || [])].slice(-4).reverse(),
    [runReplay],
  );

  const renderRunMap = (ledger: Record<string, unknown>) => {
    const summary =
      ledger.summary && typeof ledger.summary === 'object'
        ? (ledger.summary as Record<string, unknown>)
        : {};
    const planSteps = ledgerList(ledger.plan_steps);
    const pendingSteps = ledgerList(ledger.pending_steps);
    const toolCalls = ledgerList(ledger.tool_calls);
    const toolResults = ledgerList(ledger.tool_results);
    const evidenceItems = ledgerList(ledger.evidence_items);
    const memoryWrites = ledgerList(ledger.memory_writes);
    const missing = Array.isArray(ledger.missing_requirements)
      ? ledger.missing_requirements.map(String)
      : [];
    const verification =
      ledger.verification && typeof ledger.verification === 'object'
        ? (ledger.verification as Record<string, unknown>)
        : null;

    return (
      <div className='run-map'>
        <div className='run-map-head'>
          <div>
            <div className='run-map-title'>Run Map</div>
            <div className='run-map-subtitle'>
              {String(ledger.objective || 'objective not recorded')}
            </div>
          </div>
          <div className='run-map-pills'>
            <span>{String(ledger.ledger_mode || 'shadow')}</span>
            <span>{String(ledger.phase || 'phase')}</span>
            <span>{Number(summary.event_count || 0)} events</span>
          </div>
        </div>
        <div className='run-map-grid'>
          <div className='run-map-card'>
            <span>Plan</span>
            <strong>{planSteps.length} steps</strong>
            <small>{pendingSteps.length} pending</small>
          </div>
          <div className='run-map-card'>
            <span>Tools</span>
            <strong>{toolCalls.length} calls</strong>
            <small>{toolResults.length} results</small>
          </div>
          <div className='run-map-card'>
            <span>Evidence</span>
            <strong>{evidenceItems.length} selected</strong>
            <small>{missing.length ? `${missing.length} missing` : 'complete enough'}</small>
          </div>
          <div className='run-map-card'>
            <span>Memory</span>
            <strong>{memoryWrites.length} writes</strong>
            <small>{memoryWrites[0]?.status ? String(memoryWrites[0].status) : 'not committed'}</small>
          </div>
          <div className='run-map-card'>
            <span>Verification</span>
            <strong>{verification ? String(verification.status || 'recorded') : 'not recorded'}</strong>
            <small>{verification ? String(verification.verdict || '') : 'awaiting response'}</small>
          </div>
        </div>
        {toolResults.length > 0 && (
          <div className='run-map-list'>
            <div className='run-map-list-title'>Linked Tool Results</div>
            {toolResults.slice(-4).map((item) => (
              <div key={String(item.id)} className='run-map-row'>
                <span>{String(item.tool_name || 'tool')}</span>
                <strong>{String(item.status || 'status')}</strong>
                <small>{String(item.summary || '').slice(0, 160)}</small>
              </div>
            ))}
          </div>
        )}
        {missing.length > 0 && (
          <div className='run-map-missing'>
            missing: {missing.join(', ')}
          </div>
        )}
      </div>
    );
  };

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

          <label className='trace-control compact'>
            <span>phase</span>
            <select
              value={phaseFilter}
              onChange={(e) => setPhaseFilter(e.target.value)}
            >
              {phaseOptions.map((phase) => (
                <option key={phase} value={phase}>
                  {phase}
                </option>
              ))}
            </select>
          </label>

          <label className='trace-control compact'>
            <span>status</span>
            <select
              value={severityFilter}
              onChange={(e) =>
                setSeverityFilter(e.target.value as TraceSeverity)
              }
            >
              <option value='all'>all</option>
              <option value='info'>info</option>
              <option value='success'>success</option>
              <option value='warning'>warning</option>
              <option value='error'>error</option>
              <option value='retrying'>retrying</option>
              <option value='blocked'>blocked</option>
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
        <div className='trace-operator-view'>
          <section className='trace-operator-hero'>
            <div className='trace-operator-main'>
              <div className='trace-operator-kicker'>Operator View</div>
              <h2>{operatorStory?.objective || overview.primary[0]?.summary || 'Waiting for run objective.'}</h2>
              <p>
                {operatorStory?.next_best_action ||
                  overview.primary[1]?.summary ||
                  'No next action has been recorded yet.'}
              </p>
            </div>
            <div className='trace-operator-metrics'>
              <div>
                <span>Status</span>
                <strong>{operatorStory?.status || runReplay?.run?.status || 'idle'}</strong>
              </div>
              <div>
                <span>Tokens</span>
                <strong>{formatTokens(operatorStory?.cost?.total_tokens ?? runReplay?.summary?.total_tokens ?? 0)}</strong>
              </div>
              <div>
                <span>Cost</span>
                <strong>{formatCurrency(operatorStory?.cost?.estimated_cost_usd ?? runReplay?.summary?.estimated_cost_usd ?? 0)}</strong>
              </div>
            </div>
          </section>

          <section className='trace-lane-rail' aria-label='Run workflow lanes'>
            {storyLanes.map((lane, index) => (
              <button
                key={lane.id}
                className={`trace-lane-card state-${eventStatusClass(lane.status)}`}
                onClick={() => {
                  if (lane.event_id) {
                    setSelectedId(lane.event_id);
                    setFocusMode('inspect');
                  }
                }}
                disabled={!lane.event_id}
              >
                <div className='trace-lane-index'>{String(index + 1).padStart(2, '0')}</div>
                <div className='trace-lane-body'>
                  <div className='trace-lane-title'>
                    <span>{lane.label}</span>
                    <small>{lane.count || 0}</small>
                  </div>
                  <div className='trace-lane-summary'>{lane.summary || 'Not recorded yet.'}</div>
                  <div className='trace-lane-meta'>
                    <span>{lane.status || 'idle'}</span>
                    {lane.event_type ? <span>{humanizeShovsTraceEvent(lane.event_type)}</span> : null}
                  </div>
                </div>
              </button>
            ))}
          </section>

          <section className='trace-operator-bottom'>
            <div className='trace-operator-panel'>
              <div className='trace-section-head compact'>
                <div className='trace-section-title'>Latest Decisions</div>
                <button className='trace-action' onClick={() => setFocusMode('inspect')}>
                  inspect all
                </button>
              </div>
              <div className='trace-decision-list'>
                {runStoryCards.length === 0 ? (
                  <div className='trace-empty'>No readable decisions yet.</div>
                ) : (
                  runStoryCards.slice(0, 5).map((card) => (
                    <div key={card.id} className='trace-decision-row'>
                      <span>{card.eyebrow}</span>
                      <strong>{card.title}</strong>
                      <p>{card.summary}</p>
                    </div>
                  ))
                )}
              </div>
            </div>
            <div className='trace-operator-panel narrow'>
              <div className='trace-section-title'>Signal</div>
              <div className='trace-signal-stack'>
                {overview.secondary.length === 0 ? (
                  <div className='trace-empty'>No warnings or interventions.</div>
                ) : (
                  overview.secondary.slice(0, 3).map((card) => (
                    <div key={card.id} className={`trace-signal-card tone-${card.tone}`}>
                      <span>{card.eyebrow || card.title}</span>
                      <strong>{card.title}</strong>
                      <p>{card.summary}</p>
                    </div>
                  ))
                )}
              </div>
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
                    className={`trace-row status-${traceSeverity(event)} ${selectedId === event.id ? 'active' : ''}`}
                    onClick={() => setSelectedId(event.id)}
                  >
                    <div className='trace-row-top'>
                      <span className='t-type'>
                        {humanizeShovsTraceEvent(event.event_type)}
                      </span>
                      <span className='t-time'>{formatTime(event.ts)}</span>
                    </div>
                    <div className='trace-row-meta'>
                      <span>{traceSeverity(event)}</span>
                      <span>{tracePhase(event)}</span>
                      <span>
                        pass{' '}
                        {typeof event.pass_index === 'number'
                          ? event.pass_index
                          : '--'}
                      </span>
                      <span>{formatBytes(event.size_bytes || 0)}</span>
                      <span>{event.payload_ref ? 'blob' : 'inline'}</span>
                    </div>
                    <div className='trace-row-meta related'>
                      {relatedTraceIds(event).map((item) => (
                        <span key={item}>{item}</span>
                      ))}
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
                  <div className='trace-detail-main'>
                    <div className='trace-detail-title'>
                      {humanizeShovsTraceEvent(selectedEvent.event_type)}
                    </div>
                    <div className='trace-detail-actions'>
                      <button
                        className='trace-action'
                        onClick={() => void copyText(selectedSummary)}
                      >
                        copy summary
                      </button>
                      <button
                        className='trace-action'
                        onClick={() => void copyText(eventJson || '{}')}
                      >
                        copy JSON
                      </button>
                      {selectedEvent.run_id ? (
                        <button
                          className='trace-action'
                          onClick={() => setSearch(selectedEvent.run_id || '')}
                        >
                          jump to run
                        </button>
                      ) : null}
                    </div>
                  </div>
                  <div className='trace-detail-meta'>
                    <span>status: {traceSeverity(selectedEvent)}</span>
                    <span>phase: {tracePhase(selectedEvent)}</span>
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

                {selectedRunLedger && (
                  <div className='trace-run-map-wrap'>
                    {renderRunMap(selectedRunLedger)}
                  </div>
                )}

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

                        {(runReplay?.evals?.length || 0) > 0 && (
                          <div className='trace-run-list'>
                            {runReplay!
                              .evals!.slice(-4)
                              .reverse()
                              .map((item) => (
                                <div
                                  key={item.eval_id}
                                  className={`trace-run-card ${item.passed ? 'tone-good' : 'tone-warn'}`}
                                >
                                  <div className='trace-run-card-head'>
                                    <span>
                                      {item.eval_type.replace(/_/g, ' ')}
                                    </span>
                                    <span>
                                      {item.phase}
                                      {typeof item.score === 'number'
                                        ? ` · ${Math.round(item.score * 100)}%`
                                        : ''}
                                    </span>
                                  </div>
                                  <div className='trace-run-card-copy'>
                                    {item.detail || 'No evaluation detail.'}
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
