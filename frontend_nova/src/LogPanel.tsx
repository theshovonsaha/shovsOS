import React, { useState, useEffect, useRef, useCallback } from 'react';

// ── Types ────────────────────────────────────────────────────────────────────

interface LogEntry {
    ts: number;
    category: 'agent' | 'tool' | 'rag' | 'llm' | 'ctx' | 'system';
    session: string;
    message: string;
    level: 'info' | 'ok' | 'warn' | 'error';
    meta: Record<string, any>;
}

interface LogPanelProps {
    sessionId?: string | null;
    isOpen: boolean;
    onClose: () => void;
}

// ── Constants ────────────────────────────────────────────────────────────────

const CATEGORIES = ['all', 'agent', 'tool', 'rag', 'llm', 'ctx', 'system'] as const;
type CategoryFilter = typeof CATEGORIES[number];

const CAT_COLOR: Record<string, string> = {
    agent: '#00e87a',
    tool: '#ffb300',
    rag: '#00b8ff',
    llm: '#c084fc',
    ctx: '#fb923c',
    system: '#6b7280',
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

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatTime(ts: number): string {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
        + '.' + String(d.getMilliseconds()).padStart(3, '0');
}

function shortSession(session: string): string {
    if (session === 'system') return 'sys';
    return session.slice(0, 6);
}

// ── Main Component ────────────────────────────────────────────────────────────

export const LogPanel: React.FC<LogPanelProps> = ({ sessionId, isOpen, onClose }) => {
    const [entries, setEntries] = useState<LogEntry[]>([]);
    const [filter, setFilter] = useState<CategoryFilter>('all');
    const [search, setSearch] = useState('');
    const [paused, setPaused] = useState(false);
    const [connected, setConnected] = useState(false);
    const [autoScroll, setAutoScroll] = useState(true);

    const bottomRef = useRef<HTMLDivElement>(null);
    const entriesRef = useRef<HTMLDivElement>(null);
    const esRef = useRef<EventSource | null>(null);
    const pauseRef = useRef(paused);
    pauseRef.current = paused;

    // Connect to log stream
    useEffect(() => {
        if (!isOpen) return;

        const url = sessionId
            ? `/api/logs/stream?session_id=${sessionId}`
            : '/api/logs/stream';

        const es = new EventSource(url);
        esRef.current = es;

        es.onopen = () => setConnected(true);
        es.onerror = () => setConnected(false);

        es.onmessage = (e) => {
            if (pauseRef.current) return;
            try {
                const entry: LogEntry = JSON.parse(e.data);
                setEntries(prev => {
                    const next = [...prev, entry];
                    return next.length > MAX_ENTRIES ? next.slice(-MAX_ENTRIES) : next;
                });
            } catch { }
        };

        return () => { es.close(); setConnected(false); };
    }, [isOpen, sessionId]);

    // Auto-scroll
    useEffect(() => {
        if (autoScroll && !paused) {
            const container = entriesRef.current;
            if (!container) return;
            container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
        }
    }, [entries, autoScroll, paused]);

    const clearLogs = useCallback(() => setEntries([]), []);

    const filtered = entries.filter(e => {
        if (filter !== 'all' && e.category !== filter) return false;
        if (search && !e.message.toLowerCase().includes(search.toLowerCase())
            && !e.session.includes(search)) return false;
        return true;
    });

    if (!isOpen) return null;

    return (
        <div className="log-panel">
            {/* Header */}
            <div className="log-header">
                <div className="log-title">
                    <span className="log-title-glyph">⬡</span>
                    <span>INTERNAL LOG</span>
                    <span className={`log-conn ${connected ? 'live' : 'dead'}`}>
                        ● {connected ? 'live' : 'disconnected'}
                    </span>
                </div>
                <div className="log-header-actions">
                    <button
                        className={`log-btn ${paused ? 'active' : ''}`}
                        onClick={() => setPaused(p => !p)}
                        title={paused ? 'Resume' : 'Pause'}
                    >
                        {paused ? '▶ resume' : '⏸ pause'}
                    </button>
                    <button className="log-btn" onClick={clearLogs} title="Clear">
                        ⊘ clear
                    </button>
                    <button className="log-close" onClick={onClose}>✕</button>
                </div>
            </div>

            {/* Filter bar */}
            <div className="log-filters">
                {CATEGORIES.map(cat => (
                    <button
                        key={cat}
                        className={`log-cat-btn ${filter === cat ? 'active' : ''}`}
                        onClick={() => setFilter(cat)}
                        style={filter === cat && cat !== 'all'
                            ? { borderColor: CAT_COLOR[cat], color: CAT_COLOR[cat] }
                            : {}}
                    >
                        {cat !== 'all' && (
                            <span className="log-cat-dot" style={{ background: CAT_COLOR[cat] }} />
                        )}
                        {cat}
                    </button>
                ))}
                <div className="log-search-wrap">
                    <input
                        className="log-search"
                        placeholder="filter…"
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                    />
                </div>
                <span className="log-count">{filtered.length}</span>
            </div>

            {/* Entries */}
            <div
                ref={entriesRef}
                className="log-entries"
                onScroll={e => {
                    const el = e.currentTarget;
                    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
                    setAutoScroll(atBottom);
                }}
            >
                {filtered.length === 0 ? (
                    <div className="log-empty">
                        <div>◈</div>
                        <div>no log entries{filter !== 'all' ? ` for [${filter}]` : ''}</div>
                    </div>
                ) : (
                    filtered.map((entry, i) => (
                        <LogRow key={`${entry.ts}-${i}`} entry={entry} />
                    ))
                )}
                <div ref={bottomRef} />
            </div>

            {/* Footer */}
            <div className="log-footer">
                <span>{entries.length} total · {filtered.length} shown</span>
                {!autoScroll && (
                    <button
                        className="log-btn"
                        onClick={() => {
                            const container = entriesRef.current;
                            setAutoScroll(true);
                            if (container) container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
                        }}
                    >
                        ↓ scroll to bottom
                    </button>
                )}
            </div>
        </div>
    );
};


// ── Log Row ──────────────────────────────────────────────────────────────────

const LogRow: React.FC<{ entry: LogEntry }> = React.memo(({ entry }) => {
    const [expanded, setExpanded] = useState(false);
    const hasMeta = entry.meta && Object.keys(entry.meta).length > 0;

    return (
        <div
            className={`log-row level-${entry.level} ${expanded ? 'expanded' : ''}`}
            onClick={() => hasMeta && setExpanded(e => !e)}
            style={{ cursor: hasMeta ? 'pointer' : 'default' }}
        >
            <div className="log-ts">{formatTime(entry.ts)}</div>
            <div
                className="log-cat"
                style={{ color: CAT_COLOR[entry.category] || 'var(--text-dim)' }}
            >
                {entry.category}
            </div>
            <div className="log-session" title={entry.session}>
                {shortSession(entry.session)}
            </div>
            <div
                className="log-glyph"
                style={{ color: LEVEL_COLOR[entry.level] }}
            >
                {LEVEL_GLYPH[entry.level]}
            </div>
            <div
                className="log-msg"
                style={{ color: entry.level === 'info' ? 'var(--text)' : LEVEL_COLOR[entry.level] }}
            >
                {entry.message}
            </div>
            {hasMeta && (
                <div
                    className="log-meta-hint"
                    style={{
                        transform: expanded ? 'rotate(90deg)' : 'none',
                        transition: 'transform 0.2s cubic-bezier(0.16, 1, 0.3, 1)'
                    }}
                >
                    ›
                </div>
            )}

            {expanded && hasMeta && (
                <div className="log-meta" onClick={e => e.stopPropagation()}>
                    {Object.entries(entry.meta).map(([k, v]) => (
                        <div key={k} className="log-meta-row">
                            <span className="log-meta-key">{k}</span>
                            <span className="log-meta-val">
                                {typeof v === 'object' ? (
                                    <pre style={{ margin: 0, background: 'transparent', padding: 0 }}>
                                        {JSON.stringify(v, null, 2)}
                                    </pre>
                                ) : String(v)}
                            </span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
});

LogRow.displayName = 'LogRow';
