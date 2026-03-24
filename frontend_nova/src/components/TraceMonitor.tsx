import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { getOwnerId } from '../owner';

interface TraceEventSummary {
  id: string;
  ts: number;
  iso_ts?: string;
  agent_id: string;
  session_id: string;
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

interface TraceMonitorProps {
  sessionId?: string | null;
  isVisible: boolean;
}

const TRACE_PAGE_SIZE = 120;
const VIEW_SCOPE_KEY = 'shovs_trace_scope';
const EVENT_FILTER_KEY = 'shovs_trace_filter';
const AUTO_REFRESH_KEY = 'shovs_trace_auto';

const BASE_EVENT_TYPES = [
  'story',
  'all',
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
  'plan',
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

const HUMAN_EVENT_LABELS: Record<string, string> = {
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

function describeTraceEvent(event: TraceEventSummary): string {
  const data = (event.data && typeof event.data === 'object' ? event.data : {}) as Record<string, any>;
  switch (event.event_type) {
    case 'runtime_loop_mode':
      return `Loop mode: ${data.effective || data.requested || 'unknown'}`;
    case 'route_decision':
      return `Route: ${data.route_type || 'unknown'}`;
    case 'plan':
      return String(data.strategy || event.preview || 'Planner issued a strategy.');
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
    case 'plan':
      return 'Plan';
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
      typeof event.pass_index === 'number' ? `pass-${event.pass_index}` : `event-${event.id}`;
    const bucket = groups.get(key) || [];
    bucket.push(event);
    groups.set(key, bucket);
  }

  return Array.from(groups.entries())
    .map(([key, bucket]) => {
      const passIndex = bucket.find((event) => typeof event.pass_index === 'number')?.pass_index;
      const stageLabels = Array.from(
        new Set(
          bucket
            .map((event) => labelForStoryStage(event.event_type))
            .filter((value): value is string => Boolean(value)),
        ),
      );
      const lines = bucket.map((event) => describeTraceEvent(event)).filter(Boolean);
      const responseEvent =
        [...bucket].reverse().find((event) => event.event_type === 'assistant_response') ||
        [...bucket].reverse().find((event) => event.event_type === 'verification_warning') ||
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
}) => {
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

  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
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

    fetch(`/api/logs/traces/event/${encodeURIComponent(selectedId)}`, {
      signal: controller.signal,
    })
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

      {error && <div className='trace-error'>{error}</div>}

      <div className='trace-grid'>
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
                      <span className='trace-story-pass'>{group.passLabel}</span>
                      <span className='trace-story-time'>{formatTime(group.startedAt)}</span>
                    </div>
                    <div className='trace-story-headline'>{group.headline}</div>
                    <div className='trace-story-badges'>
                      {group.stageLabels.map((label) => (
                        <span key={`${group.id}-${label}`} className='trace-story-badge'>
                          {label}
                        </span>
                      ))}
                    </div>
                    <div className='trace-story-lines'>
                      {group.lines.map((line, index) => (
                        <div key={`${group.id}-line-${index}`} className='trace-story-line'>
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
                      <span className='t-type'>{humanizeEventType(event.event_type)}</span>
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
                    <div className='trace-row-preview'>{describeTraceEvent(event)}</div>
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
      </div>
    </div>
  );
};
