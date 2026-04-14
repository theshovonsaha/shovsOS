import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  buildTimelineEntries as buildNovaTimelineEntries,
  describeTraceEvent as describeNovaTraceEvent,
} from './monitor/presentation';
import { getOwnerId } from './owner';
import { OperatorInterventions } from './components/OperatorInterventions';

interface LogEntry {
  ts: number;
  category: string;
  session: string;
  message: string;
  level: 'info' | 'ok' | 'warn' | 'error';
  owner_id?: string | null;
  meta: Record<string, any>;
}

interface LogPanelProps {
  sessionId?: string | null;
  isOpen: boolean;
  onClose: () => void;
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

interface RecentLogsResponse {
  logs?: LogEntry[];
}

interface TraceEventSummary {
  id: string;
  ts: number;
  event_type: string;
  preview?: string;
  run_id?: string | null;
  pass_index?: number | null;
  data?: Record<string, any> | null;
}

interface TraceRecentResponse {
  events?: TraceEventSummary[];
}

interface ConsoleCard {
  id: string;
  title: string;
  eyebrow: string;
  summary: string;
  detail?: string;
  tone: 'neutral' | 'good' | 'warn';
}

interface TimelineEntry {
  id: string;
  stage: string;
  title: string;
  summary: string;
  toolName?: string;
  ts: number;
  tone: 'neutral' | 'good' | 'warn';
}

const PREFERRED_CATEGORIES = [
  'agent',
  'tool',
  'rag',
  'llm',
  'ctx',
  'system',
] as const;

const CAT_COLOR: Record<string, string> = {
  agent: '#00e87a',
  tool: '#ffb300',
  rag: '#00b8ff',
  llm: '#c084fc',
  ctx: '#fb923c',
  system: '#94a3b8',
};

const LEVEL_COLOR: Record<string, string> = {
  info: 'var(--text-dim)',
  ok: '#00e87a',
  warn: '#ffb300',
  error: '#ff4444',
};

const LEVEL_GLYPH: Record<string, string> = {
  info: '·',
  ok: '✓',
  warn: '!',
  error: '✗',
};

const MAX_ENTRIES = 300;

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  return (
    d.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    }) +
    '.' +
    String(d.getMilliseconds()).padStart(3, '0')
  );
}

function shortSession(session: string): string {
  if (session === 'system') return 'sys';
  return session.slice(0, 6);
}

function mergeEntries(existing: LogEntry[], incoming: LogEntry[]): LogEntry[] {
  const keyed = new Map<string, LogEntry>();
  for (const entry of [...existing, ...incoming]) {
    const key = `${entry.ts}:${entry.category}:${entry.session}:${entry.message}`;
    keyed.set(key, entry);
  }
  return Array.from(keyed.values())
    .sort((a, b) => a.ts - b.ts)
    .slice(-MAX_ENTRIES);
}

function categoryLabel(category: string): string {
  return category;
}

function clipText(text: string, max = 140): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max - 3).trimEnd()}...`;
}

function findLatestTrace(
  events: TraceEventSummary[],
  eventTypes: string[],
): TraceEventSummary | null {
  for (const event of events) {
    if (eventTypes.includes(event.event_type)) return event;
  }
  return null;
}

function buildConsoleCards(traceEvents: TraceEventSummary[]): ConsoleCard[] {
  const runtimeEvent = findLatestTrace(traceEvents, [
    'route_decision',
    'runtime_loop_mode',
    'phase_context',
  ]);
  const toolEvent = findLatestTrace(traceEvents, ['tool_result', 'tool_call']);
  const verifyEvent = findLatestTrace(traceEvents, [
    'verification_warning',
    'verification_result',
  ]);
  const responseEvent = findLatestTrace(traceEvents, ['assistant_response']);

  const cards: ConsoleCard[] = [];

  if (runtimeEvent) {
    cards.push({
      id: `runtime:${runtimeEvent.id}`,
      title: 'Runtime',
      eyebrow: runtimeEvent.event_type.replace(/_/g, ' '),
      summary: clipText(describeNovaTraceEvent(runtimeEvent), 150),
      detail: runtimeEvent.run_id ? `run ${runtimeEvent.run_id.slice(0, 10)}` : undefined,
      tone: runtimeEvent.event_type === 'route_decision' ? 'good' : 'neutral',
    });
  }

  if (toolEvent) {
    cards.push({
      id: `tool:${toolEvent.id}`,
      title: 'Latest Tool',
      eyebrow: typeof toolEvent.pass_index === 'number' ? `pass ${toolEvent.pass_index}` : 'tool activity',
      summary: clipText(describeNovaTraceEvent(toolEvent), 150),
      detail: toolEvent.event_type === 'tool_result' ? 'observe phase' : 'act phase',
      tone:
        toolEvent.event_type === 'tool_result' &&
        toolEvent.data?.success === false
          ? 'warn'
          : 'neutral',
    });
  }

  if (verifyEvent) {
    cards.push({
      id: `verify:${verifyEvent.id}`,
      title: 'Verification',
      eyebrow: verifyEvent.event_type.replace(/_/g, ' '),
      summary: clipText(describeNovaTraceEvent(verifyEvent), 150),
      tone:
        verifyEvent.event_type === 'verification_warning' ||
        verifyEvent.data?.supported === false
          ? 'warn'
          : 'good',
    });
  }

  if (responseEvent) {
    cards.push({
      id: `response:${responseEvent.id}`,
      title: 'Response',
      eyebrow: 'latest output',
      summary: clipText(describeNovaTraceEvent(responseEvent), 150),
      tone: 'neutral',
    });
  }

  return cards;
}

function buildTimelineEntries(traceEvents: TraceEventSummary[]): TimelineEntry[] {
  return buildNovaTimelineEntries(traceEvents, 18).map((entry) => ({
    id: entry.id,
    stage: entry.stage,
    title: entry.headline,
    summary: clipText(entry.lines.join(' '), 180),
    toolName: entry.toolName,
    ts: entry.ts,
    tone: entry.tone,
  }));
}

export const LogPanel: React.FC<LogPanelProps> = ({
  sessionId,
  isOpen,
  onClose,
  isStreaming,
  pendingConfirmation,
  onApproveConfirmation,
  onDenyConfirmation,
  onStopExecution,
}) => {
  const ownerId = useMemo(() => getOwnerId(), []);
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [traceEvents, setTraceEvents] = useState<TraceEventSummary[]>([]);
  const [view, setView] = useState<'overview' | 'timeline' | 'logs'>('overview');
  const [filter, setFilter] = useState<string>('all');
  const [search, setSearch] = useState('');
  const [paused, setPaused] = useState(false);
  const [connected, setConnected] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const bottomRef = useRef<HTMLDivElement>(null);
  const entriesRef = useRef<HTMLDivElement>(null);
  const esRef = useRef<EventSource | null>(null);
  const pauseRef = useRef(paused);
  pauseRef.current = paused;

  const fetchTraceConsole = useCallback(async () => {
    const params = new URLSearchParams();
    params.set('limit', '28');
    params.set('owner_id', ownerId);
    if (sessionId) params.set('session_id', sessionId);

    const res = await fetch(`/api/logs/traces/recent?${params.toString()}`);
    if (!res.ok) {
      throw new Error(`Trace console failed: HTTP ${res.status}`);
    }
    const data: TraceRecentResponse = await res.json();
    setTraceEvents(Array.isArray(data.events) ? data.events : []);
  }, [ownerId, sessionId]);

  useEffect(() => {
    if (!isOpen) return;

    const historyParams = new URLSearchParams();
    historyParams.set('limit', '150');
    historyParams.set('owner_id', ownerId);
    if (sessionId) historyParams.set('session_id', sessionId);

    setLoadingHistory(true);
    setLoadError(null);

    Promise.all([
      fetch(`/api/logs/recent?${historyParams.toString()}`).then(async (res) => {
        if (!res.ok) throw new Error(`Recent logs failed: HTTP ${res.status}`);
        const data: RecentLogsResponse = await res.json();
        const incoming = Array.isArray(data.logs) ? data.logs : [];
        setEntries((prev) => mergeEntries(prev, incoming));
      }),
      fetchTraceConsole(),
    ])
      .catch((err) => {
        setLoadError(
          err instanceof Error ? err.message : 'Failed to load operator history',
        );
      })
      .finally(() => setLoadingHistory(false));

    const streamParams = new URLSearchParams();
    streamParams.set('owner_id', ownerId);
    if (sessionId) streamParams.set('session_id', sessionId);

    const es = new EventSource(`/api/logs/stream?${streamParams.toString()}`);
    esRef.current = es;

    es.onopen = () => setConnected(true);
    es.onerror = () => {
      setConnected(false);
      setLoadError((prev) => prev || 'Live log stream disconnected');
    };
    es.onmessage = (e) => {
      if (pauseRef.current) return;
      try {
        const entry: LogEntry = JSON.parse(e.data);
        setLoadError(null);
        setEntries((prev) => mergeEntries(prev, [entry]));
      } catch {
        // Ignore malformed SSE rows.
      }
    };

    const traceInterval = window.setInterval(() => {
      fetchTraceConsole().catch((err) => {
        setLoadError(
          err instanceof Error ? err.message : 'Trace console refresh failed',
        );
      });
    }, 3000);

    return () => {
      es.close();
      window.clearInterval(traceInterval);
      setConnected(false);
    };
  }, [fetchTraceConsole, isOpen, ownerId, sessionId]);

  useEffect(() => {
    if (autoScroll && !paused) {
      const container = entriesRef.current;
      if (!container) return;
      container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
    }
  }, [entries, autoScroll, paused]);

  const clearLogs = useCallback(() => setEntries([]), []);

  const categories = useMemo(() => {
    const discovered = new Set<string>();
    for (const entry of entries) discovered.add(entry.category);

    const ordered = [
      ...PREFERRED_CATEGORIES.filter((cat) => discovered.has(cat)),
      ...Array.from(discovered)
        .filter((cat) => !PREFERRED_CATEGORIES.includes(cat as (typeof PREFERRED_CATEGORIES)[number]))
        .sort(),
    ];

    return ['all', ...ordered];
  }, [entries]);

  const consoleCards = useMemo(() => buildConsoleCards(traceEvents), [traceEvents]);
  const timelineEntries = useMemo(
    () => buildTimelineEntries(traceEvents),
    [traceEvents],
  );

  const filtered = entries.filter((entry) => {
    if (filter !== 'all' && entry.category !== filter) return false;
    if (!search) return true;

    const needle = search.toLowerCase();
    const haystack = [
      entry.message,
      entry.session,
      entry.category,
      entry.meta?.source_category || '',
      entry.meta?.tool_name || '',
    ]
      .join(' ')
      .toLowerCase();

    return haystack.includes(needle);
  });

  if (!isOpen) return null;

  return (
    <div className='log-panel'>
      <div className='log-header'>
        <div className='log-title'>
          <span className='log-title-glyph'>⬡</span>
          <span>System Console</span>
          <span className={`log-conn ${connected ? 'live' : 'dead'}`}>
            ● {connected ? 'live' : 'disconnected'}
          </span>
        </div>
        <div className='log-header-actions'>
          <button
            className={`log-btn ${paused ? 'active' : ''}`}
            onClick={() => setPaused((p) => !p)}
            title={paused ? 'Resume' : 'Pause'}
          >
            {paused ? '▶ resume' : '⏸ pause'}
          </button>
          <button className='log-btn' onClick={clearLogs} title='Clear'>
            ⊘ clear
          </button>
          <button className='log-close' onClick={onClose}>
            ✕
          </button>
        </div>
      </div>

      {loadError && <div className='trace-error'>{loadError}</div>}

      <OperatorInterventions
        sessionId={sessionId}
        isStreaming={isStreaming}
        pendingConfirmation={pendingConfirmation}
        onApprove={onApproveConfirmation}
        onDeny={onDenyConfirmation}
        onStop={onStopExecution}
      />

      <div className='log-console'>
        {consoleCards.length === 0 ? (
          <div className='log-console-empty'>
            Waiting for structured runtime traces...
          </div>
        ) : (
          consoleCards.map((card) => (
            <div
              key={card.id}
              className={`log-console-card tone-${card.tone}`}
            >
              <div className='log-console-top'>
                <span className='log-console-title'>{card.title}</span>
                <span className='log-console-eyebrow'>{card.eyebrow}</span>
              </div>
              <div className='log-console-summary'>{card.summary}</div>
              {card.detail ? (
                <div className='log-console-detail'>{card.detail}</div>
              ) : null}
            </div>
          ))
        )}
      </div>

      <div className='log-view-switch'>
        {(['overview', 'timeline', 'logs'] as const).map((name) => (
          <button
            key={name}
            className={view === name ? 'active' : ''}
            onClick={() => setView(name)}
          >
            {name}
          </button>
        ))}
      </div>

      {view === 'logs' ? (
      <div className='log-filters'>
        {categories.map((cat) => (
          <button
            key={cat}
            className={`log-cat-btn ${filter === cat ? 'active' : ''}`}
            onClick={() => setFilter(cat)}
            style={
              filter === cat && cat !== 'all'
                ? { borderColor: CAT_COLOR[cat] || 'var(--border-hi)', color: CAT_COLOR[cat] || 'var(--text)' }
                : {}
            }
          >
            {cat !== 'all' && (
              <span
                className='log-cat-dot'
                style={{ background: CAT_COLOR[cat] || 'var(--text-dim)' }}
              />
            )}
            {cat === 'all' ? 'all' : categoryLabel(cat)}
          </button>
        ))}
        <div className='log-search-wrap'>
          <input
            className='log-search'
            placeholder='search logs...'
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <span className='log-count'>{filtered.length}</span>
      </div>
      ) : null}

      <div
        ref={entriesRef}
        className='log-entries'
        onScroll={(e) => {
          const el = e.currentTarget;
          const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
          setAutoScroll(atBottom);
        }}
      >
        {loadingHistory && entries.length === 0 ? (
          <div className='log-empty'>
            <div>◈</div>
            <div>loading operator history...</div>
          </div>
        ) : view === 'overview' ? (
          <div className='log-overview-grid'>
            <div className='log-overview-card'>
              <div className='log-overview-title'>Categories</div>
              <div className='log-overview-tags'>
                {categories
                  .filter((cat) => cat !== 'all')
                  .map((cat) => (
                    <span key={cat} className='log-overview-tag'>
                      <span
                        className='log-cat-dot'
                        style={{ background: CAT_COLOR[cat] || 'var(--text-dim)' }}
                      />
                      {cat}
                    </span>
                  ))}
              </div>
            </div>
            <div className='log-overview-card'>
              <div className='log-overview-title'>Latest phases</div>
              <div className='log-phase-list'>
                {timelineEntries.slice(0, 6).map((entry) => (
                  <div key={entry.id} className={`log-phase-item tone-${entry.tone}`}>
                    <span className='log-phase-stage'>{entry.stage}</span>
                    <span className='log-phase-copy'>{entry.summary}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : view === 'timeline' ? (
          timelineEntries.length === 0 ? (
            <div className='log-empty'>
              <div>◈</div>
              <div>no runtime phase activity yet</div>
            </div>
          ) : (
            <div className='log-timeline'>
              {timelineEntries.map((entry) => (
                <div key={entry.id} className={`log-timeline-card tone-${entry.tone}`}>
                  <div className='log-timeline-top'>
                    <span className='log-timeline-stage'>{entry.stage}</span>
                    <span className='log-timeline-time'>{formatTime(entry.ts)}</span>
                  </div>
                  <div className='log-timeline-title'>{entry.title}</div>
                  <div className='log-timeline-summary'>{entry.summary}</div>
                  {entry.toolName ? (
                    <div className='log-timeline-tool'>{entry.toolName}</div>
                  ) : null}
                </div>
              ))}
            </div>
          )
        ) : filtered.length === 0 ? (
          <div className='log-empty'>
            <div>◈</div>
            <div>no log entries{filter !== 'all' ? ` for [${filter}]` : ''}</div>
          </div>
        ) : (
          filtered.map((entry, index) => (
            <LogRow key={`${entry.ts}-${index}`} entry={entry} />
          ))
        )}
        <div ref={bottomRef} />
      </div>

      <div className='log-footer'>
        <span>{entries.length} total · {filtered.length} shown</span>
        {!autoScroll && (
          <button
            className='log-btn'
            onClick={() => {
              const container = entriesRef.current;
              setAutoScroll(true);
              if (container) {
                container.scrollTo({
                  top: container.scrollHeight,
                  behavior: 'smooth',
                });
              }
            }}
          >
            ↓ scroll to bottom
          </button>
        )}
      </div>
    </div>
  );
};

const LogRow: React.FC<{ entry: LogEntry }> = React.memo(({ entry }) => {
  const [expanded, setExpanded] = useState(false);
  const hasMeta = entry.meta && Object.keys(entry.meta).length > 0;
  const categoryDisplay = categoryLabel(entry.category);
  const sourceCategory = String(entry.meta?.source_category || '').trim();
  const stageLabel = String(entry.meta?.phase || entry.meta?.trace_phase || '').trim();
  const toolName = String(entry.meta?.tool_name || entry.meta?.tool || '').trim();

  return (
    <div
      className={`log-row level-${entry.level} ${expanded ? 'expanded' : ''}`}
      onClick={() => hasMeta && setExpanded((e) => !e)}
      style={{ cursor: hasMeta ? 'pointer' : 'default' }}
    >
      <div className='log-ts'>{formatTime(entry.ts)}</div>
      <div
        className='log-cat'
        style={{ color: CAT_COLOR[entry.category] || 'var(--text-dim)' }}
      >
        {categoryDisplay}
        {sourceCategory && sourceCategory !== entry.category ? (
          <span className='log-cat-source'>{sourceCategory}</span>
        ) : null}
      </div>
      <div className='log-session' title={entry.session}>
        {shortSession(entry.session)}
      </div>
      <div
        className='log-glyph'
        style={{ color: LEVEL_COLOR[entry.level] }}
      >
        {LEVEL_GLYPH[entry.level]}
      </div>
      <div
        className='log-msg'
        style={{
          color: entry.level === 'info' ? 'var(--text)' : LEVEL_COLOR[entry.level],
        }}
      >
        {entry.message}
        {(stageLabel || toolName) && (
          <div className='log-row-tags'>
            {stageLabel ? <span className='log-row-tag'>{stageLabel}</span> : null}
            {toolName ? <span className='log-row-tag'>{toolName}</span> : null}
          </div>
        )}
      </div>
      {hasMeta && (
        <div
          className='log-meta-hint'
          style={{
            transform: expanded ? 'rotate(90deg)' : 'none',
            transition: 'transform 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
          }}
        >
          ›
        </div>
      )}

      {expanded && hasMeta && (
        <div className='log-meta' onClick={(e) => e.stopPropagation()}>
          {Object.entries(entry.meta).map(([key, value]) => (
            <div key={key} className='log-meta-row'>
              <span className='log-meta-key'>{key}</span>
              <span className='log-meta-val'>
                {typeof value === 'object' ? (
                  <pre style={{ margin: 0, background: 'transparent', padding: 0 }}>
                    {JSON.stringify(value, null, 2)}
                  </pre>
                ) : (
                  String(value)
                )}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

LogRow.displayName = 'LogRow';
