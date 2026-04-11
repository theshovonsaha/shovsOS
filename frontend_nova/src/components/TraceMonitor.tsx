import React, { useCallback, useEffect, useMemo, useState } from 'react';
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
  'story',
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

const IMPORTANT_EVENT_TYPES = new Set([
  'runtime_loop_mode',
  'route_decision',
  'conversation_tension',
  'stance_signals_extracted',
  'plan',
  'phase_context',
  'compiled_context',
  'tool_call',
  'tool_result',
  'manager_observation',
  'verification_result',
  'verification_warning',
  'assistant_response',
]);

interface StoryGroup {
  id: string;
  dominantEventId: string;
  passLabel: string;
  startedAt?: number;
  stageLabels: string[];
  headline: string;
  lines: string[];
}

interface MonitorInsight {
  id: string;
  title: string;
  summary: string;
  tone: 'neutral' | 'good' | 'warn';
}

const HUMAN_EVENT_LABELS: Record<string, string> = {
  conversation_tension: 'Conversation tension detected',
  stance_signals_extracted: 'Stance candidates extracted',
  phase_context: 'Phase packet compiled',
  compiled_context: 'Message prompt compiled',
  llm_pass_start: 'Model pass started',
  llm_prompt: 'Prompt sent to model',
  llm_pass_complete: 'Model pass completed',
  tool_call: 'Tool called',
  tool_result: 'Tool returned',
  prompt_components: 'Prompt context built',
  assistant_response: 'Assistant response finalized',
};

function humanizeEventType(eventType: string): string {
  if (eventType === 'story') return 'Readable Timeline';
  return HUMAN_EVENT_LABELS[eventType] || eventType.replace(/_/g, ' ');
}

function summarizeIncludedItems(data: Record<string, any>): {
  count: number;
  hasTension: boolean;
  preview: string;
} {
  const included = Array.isArray(data.included) ? data.included : [];
  const itemIds = included
    .map((item) =>
      item && typeof item === 'object' ? String(item.item_id || '').trim() : '',
    )
    .filter(Boolean);
  return {
    count: itemIds.length,
    hasTension: itemIds.includes('conversation_tension'),
    preview: itemIds.slice(0, 4).join(', '),
  };
}

function describeTraceEvent(event: TraceEventSummary): string {
  const data = (
    event.data && typeof event.data === 'object' ? event.data : {}
  ) as Record<string, any>;
  switch (event.event_type) {
    case 'runtime_loop_mode':
      return `Loop mode: ${data.effective || data.requested || 'unknown'}`;
    case 'route_decision':
      return `Route: ${data.route_type || 'unknown'}`;
    case 'conversation_tension': {
      const summary = String(data.summary || event.preview || '').trim();
      const challenge = String(data.challenge_level || 'low').trim();
      const conflicts = Array.isArray(data.conflicting_facts)
        ? data.conflicting_facts.length
        : 0;
      if (summary) {
        return `Tension: ${summary} · challenge=${challenge}${conflicts ? ` · conflicts=${conflicts}` : ''}`;
      }
      return `Tension detected · challenge=${challenge}${conflicts ? ` · conflicts=${conflicts}` : ''}`;
    }
    case 'stance_signals_extracted': {
      const count = Number(data.count || 0);
      const signals = Array.isArray(data.signals) ? data.signals : [];
      const preview = signals
        .slice(0, 2)
        .map((item) => String(item?.topic || item?.position || '').trim())
        .filter(Boolean)
        .join(', ');
      return `Stance candidates: ${count || signals.length}${preview ? ` · ${preview}` : ''}`;
    }
    case 'plan':
      return String(
        data.strategy || event.preview || 'Planner issued a strategy.',
      );
    case 'phase_context':
    case 'compiled_context': {
      const { count, hasTension, preview } = summarizeIncludedItems(data);
      const phase = String(data.phase || 'unknown');
      const scope = String(data.trace_scope || '').trim();
      const label =
        event.event_type === 'compiled_context'
          ? 'message prompt'
          : 'phase packet';
      const suffix = [
        `${count} item${count === 1 ? '' : 's'}`,
        hasTension ? 'tension visible' : '',
        preview ? `includes ${preview}` : '',
        scope ? `scope=${scope}` : '',
      ]
        .filter(Boolean)
        .join(' · ');
      return `${phase} ${label} compiled${suffix ? ` · ${suffix}` : ''}`;
    }
    case 'tool_call':
      return `${data.tool_name || 'tool'}: ${data.arguments_summary || 'starting'}`;
    case 'tool_result':
      return `${data.tool_name || 'tool'} ${data.success === false ? 'failed' : 'returned'}${data.content_preview ? ` · ${data.content_preview}` : ''}`;
    case 'manager_observation':
      return `${data.status || 'observation'}${data.strategy ? ` · ${data.strategy}` : ''}`;
    case 'verification_result':
      return data.supported === false
        ? `Verification flagged issues${Array.isArray(data.issues) && data.issues.length ? ` · ${data.issues[0]}` : ''}`
        : 'Verification passed';
    case 'verification_warning':
      return Array.isArray(data.issues) && data.issues.length
        ? `Warning: ${data.issues.join('; ')}`
        : 'Verification warning';
    case 'assistant_response':
      return String(data.content || event.preview || 'Final response saved.');
    default:
      return event.preview || humanizeEventType(event.event_type);
  }
}

function labelForStoryStage(eventType: string): string | null {
  switch (eventType) {
    case 'runtime_loop_mode':
      return 'Loop';
    case 'route_decision':
      return 'Route';
    case 'conversation_tension':
      return 'Tension';
    case 'stance_signals_extracted':
      return 'Tension';
    case 'plan':
      return 'Plan';
    case 'phase_context':
    case 'compiled_context':
      return 'Context';
    case 'tool_call':
      return 'Act';
    case 'tool_result':
      return 'Observe';
    case 'manager_observation':
      return 'Observe';
    case 'verification_result':
    case 'verification_warning':
      return 'Verify';
    case 'assistant_response':
      return 'Respond';
    default:
      return null;
  }
}

function buildStoryGroups(events: TraceEventSummary[]): StoryGroup[] {
  const chronological = [...events]
    .filter((event) => IMPORTANT_EVENT_TYPES.has(event.event_type))
    .sort((a, b) => a.ts - b.ts);

  const groups = new Map<string, TraceEventSummary[]>();
  for (const event of chronological) {
    const key =
      typeof event.pass_index === 'number'
        ? `pass-${event.pass_index}`
        : `event-${event.id}`;
    const bucket = groups.get(key) || [];
    bucket.push(event);
    groups.set(key, bucket);
  }

  return Array.from(groups.entries())
    .map(([key, bucket]) => {
      const passIndex = bucket.find(
        (event) => typeof event.pass_index === 'number',
      )?.pass_index;
      const stageLabels = Array.from(
        new Set(
          bucket
            .map((event) => labelForStoryStage(event.event_type))
            .filter((value): value is string => Boolean(value)),
        ),
      );
      const lines = bucket
        .map((event) => describeTraceEvent(event))
        .filter(Boolean);
      const responseEvent =
        [...bucket]
          .reverse()
          .find((event) => event.event_type === 'assistant_response') ||
        [...bucket]
          .reverse()
          .find((event) => event.event_type === 'verification_warning') ||
        bucket[bucket.length - 1];
      return {
        id: key,
        dominantEventId: responseEvent.id,
        passLabel:
          typeof passIndex === 'number'
            ? passIndex === 0
              ? 'Pass 0'
              : `Pass ${passIndex}`
            : 'Run step',
        startedAt: bucket[0]?.ts,
        stageLabels,
        headline: describeTraceEvent(responseEvent),
        lines: lines.slice(-4),
      };
    })
    .sort((a, b) => (b.startedAt || 0) - (a.startedAt || 0));
}

function latestEventOf(
  events: TraceEventSummary[],
  eventTypes: string[],
): TraceEventSummary | null {
  for (const event of events) {
    if (eventTypes.includes(event.event_type)) return event;
  }
  return null;
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

function parsePacketSections(content: string): PacketSection[] {
  const text = String(content || '').trim();
  if (!text) return [];

  const sections: PacketSection[] = [];
  const pattern = /--- (.+?) ---\n([\s\S]*?)\n--- End \1 ---/g;
  for (const match of text.matchAll(pattern)) {
    const title = String(match[1] || '').trim();
    const body = String(match[2] || '').trim();
    if (title && body) sections.push({ title, body });
  }

  if (sections.length > 0) return sections;
  return [{ title: 'Compiled Context', body: text }];
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
  const [focusMode, setFocusMode] = useState<'story' | 'inspect'>(() => {
    const stored = localStorage.getItem(TRACE_FOCUS_KEY);
    return stored === 'inspect' ? 'inspect' : 'story';
  });
  const [scope, setScope] = useState<'session' | 'all'>(() => {
    const stored = localStorage.getItem(VIEW_SCOPE_KEY);
    return stored === 'all' ? 'all' : 'session';
  });
  const [eventType, setEventType] = useState<string>(
    () => localStorage.getItem(EVENT_FILTER_KEY) || 'story',
  );
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
      eventType === 'story'
        ? events.filter((event) => IMPORTANT_EVENT_TYPES.has(event.event_type))
        : events;
    if (!search.trim()) return baseEvents;
    const needle = search.trim().toLowerCase();
    return baseEvents.filter((event) => {
      const haystack = [
        event.event_type,
        event.session_id,
        event.agent_id,
        event.preview || '',
        describeTraceEvent(event),
      ]
        .join(' ')
        .toLowerCase();
      return haystack.includes(needle);
    });
  }, [events, search]);

  const oldestTs = events.length ? events[events.length - 1].ts : undefined;
  const storyGroups = useMemo(() => buildStoryGroups(events), [events]);
  const monitorInsights = useMemo(() => {
    const runtime = latestEventOf(events, [
      'route_decision',
      'runtime_loop_mode',
      'phase_context',
    ]);
    const tool = latestEventOf(events, ['tool_result', 'tool_call']);
    const verification = latestEventOf(events, [
      'verification_warning',
      'verification_result',
    ]);
    const response = latestEventOf(events, ['assistant_response']);

    const insights: MonitorInsight[] = [];
    if (runtime) {
      insights.push({
        id: `runtime-${runtime.id}`,
        title: 'Runtime',
        summary: describeTraceEvent(runtime),
        tone: runtime.event_type === 'route_decision' ? 'good' : 'neutral',
      });
    }
    if (tool) {
      insights.push({
        id: `tool-${tool.id}`,
        title: 'Latest Tool',
        summary: describeTraceEvent(tool),
        tone:
          tool.event_type === 'tool_result' &&
          (tool.data as Record<string, unknown> | undefined)?.success === false
            ? 'warn'
            : 'neutral',
      });
    }
    if (verification) {
      insights.push({
        id: `verify-${verification.id}`,
        title: 'Verification',
        summary: describeTraceEvent(verification),
        tone:
          verification.event_type === 'verification_warning' ||
          (verification.data as Record<string, unknown> | undefined)
            ?.supported === false
            ? 'warn'
            : 'good',
      });
    }
    if (response) {
      insights.push({
        id: `response-${response.id}`,
        title: 'Response',
        summary: describeTraceEvent(response),
        tone: 'neutral',
      });
    }
    return insights;
  }, [events]);

  const fetchRecent = useCallback(
    async (opts?: { append?: boolean; beforeTs?: number }) => {
      const params = new URLSearchParams();
      params.set('limit', String(TRACE_PAGE_SIZE));
      params.set('owner_id', getOwnerId());
      if (scopedSessionId) params.set('session_id', scopedSessionId);
      if (eventType && eventType !== 'all' && eventType !== 'story')
        params.set('event_type', eventType);
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
    () => (selectedEvent ? describeTraceEvent(selectedEvent) : ''),
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
    return parsePacketSections(String(data?.content || ''));
  }, [selectedEvent]);
  const selectedIncludedItems = useMemo(() => {
    if (!selectedEvent?.data || typeof selectedEvent.data !== 'object') return [];
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
  const recentReplayPasses = useMemo(
    () => [...(runReplay?.passes || [])].slice(-4).reverse(),
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
                  {humanizeEventType(type)}
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
      </div>

      {monitorInsights.length > 0 && (
        <div className='trace-insight-row'>
          {monitorInsights.map((insight) => (
            <div
              key={insight.id}
              className={`trace-insight-card tone-${insight.tone}`}
            >
              <div className='trace-insight-title'>{insight.title}</div>
              <div className='trace-insight-summary'>{insight.summary}</div>
            </div>
          ))}
        </div>
      )}

      {error && <div className='trace-error'>{error}</div>}

      <div className={`trace-grid ${focusMode === 'story' ? 'story-focus' : 'inspect-focus'}`}>
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
            {filteredEvents.length === 0 && (
              <div className='trace-empty'>
                No trace events for this filter.
              </div>
            )}

            {eventType === 'story'
              ? storyGroups.map((group) => (
                  <button
                    key={group.id}
                    className={`trace-story-card ${selectedId === group.dominantEventId ? 'active' : ''}`}
                    onClick={() => setSelectedId(group.dominantEventId)}
                  >
                    <div className='trace-story-top'>
                      <span className='trace-story-pass'>
                        {group.passLabel}
                      </span>
                      <span className='trace-story-time'>
                        {formatTime(group.startedAt)}
                      </span>
                    </div>
                    <div className='trace-story-headline'>{group.headline}</div>
                    <div className='trace-story-badges'>
                      {group.stageLabels.map((label) => (
                        <span
                          key={`${group.id}-${label}`}
                          className='trace-story-badge'
                        >
                          {label}
                        </span>
                      ))}
                    </div>
                    <div className='trace-story-lines'>
                      {group.lines.map((line, index) => (
                        <div
                          key={`${group.id}-line-${index}`}
                          className='trace-story-line'
                        >
                          {line}
                        </div>
                      ))}
                    </div>
                  </button>
                ))
              : filteredEvents.map((event) => (
                  <button
                    key={event.id}
                    className={`trace-row ${selectedId === event.id ? 'active' : ''}`}
                    onClick={() => setSelectedId(event.id)}
                  >
                    <div className='trace-row-top'>
                      <span className='t-type'>
                        {humanizeEventType(event.event_type)}
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
                      {describeTraceEvent(event)}
                    </div>
                  </button>
                ))}
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

        {focusMode === 'inspect' ? (
        <section className='trace-detail-pane'>
          {!selectedEvent && !loadingDetail && (
            <div className='trace-empty'>
              Select an event to inspect details.
            </div>
          )}

          {loadingDetail && <div className='trace-empty'>Loading event...</div>}

          {selectedEvent && (
            <>
              <div className='trace-detail-head'>
                <div className='trace-detail-title'>
                  {humanizeEventType(selectedEvent.event_type)}
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
                        <div className='trace-packet-title'>{section.title}</div>
                        <pre className='trace-packet-copy'>{section.body}</pre>
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
                            checkpoints: {runReplay?.summary?.checkpoint_count ?? 0}
                          </span>
                          <span className='trace-run-stat'>
                            passes: {runReplay?.summary?.pass_count ?? 0}
                          </span>
                          <span className='trace-run-stat'>
                            artifacts: {runReplay?.summary?.artifact_count ?? 0}
                          </span>
                          <span className='trace-run-stat'>
                            evidence: {runReplay?.summary?.evidence_count ?? 0}
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
                        (runReplay?.latest_pass?.selected_tools || []).length >
                          0) && (
                        <div className='trace-run-head'>
                          {runReplay?.latest_checkpoint?.notes && (
                            <div className='trace-run-copy'>
                              notes: {runReplay.latest_checkpoint.notes}
                            </div>
                          )}
                          {(runReplay?.latest_pass?.selected_tools || []).length >
                            0 && (
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
                                  {pass.phase || 'phase'} · {pass.status || '--'}
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
                          {runReplay!.evidence!.slice(0, 3).map((item, index) => (
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
                          {runReplay!.artifacts!.slice(0, 4).map((artifact) => (
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
                                {(artifact.preview || '').trim() || 'No preview stored.'}
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
                      <details key={`${role}-${index}`} className='prompt-item'>
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
        ) : null}
      </div>
    </div>
  );
};
