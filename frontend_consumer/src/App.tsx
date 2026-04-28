import {
  FormEvent,
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import AgentOrb from './AgentOrb';
import MarkdownMessage from './MarkdownMessage';
import { appendOwnerId, getOwnerId } from './owner';

// ─── Types ───────────────────────────────────────────────────────────────────

type ModelsMap = Record<string, string[]>;
type Phase = 'idle' | 'thinking' | 'working' | 'finalizing';
type LayoutMode = 'centered' | 'chat';

type StreamEvent =
  | { type: 'session'; session_id: string }
  | { type: 'phase'; phase: 'thinking' | 'working' | 'finalizing' }
  | { type: 'activity_short'; text: string }
  | { type: 'activity_detail'; text: string; detail?: string }
  | { type: 'token'; content: string }
  | { type: 'done'; session_id?: string }
  | { type: 'error'; message: string }
  | { type: 'plan'; strategy: string; tools: string[]; confidence: number }
  | { type: 'verification_warning'; issues: string[]; confidence: number }
  | { type: 'redraft'; reason: string; issues: string[]; side_effect_guard: boolean }
  | { type: 'tension'; summary: string; challenge_level: string };

type StorageStatus = {
  stores: Record<string, { exists: boolean; size_bytes: number; path: string }>;
};

type MemoryStateFact = {
  subject: string;
  predicate: string;
  object: string;
  status: 'current' | 'superseded';
};

type MemorySignal = {
  event_type: string;
  label: string;
  summary: string;
  created_at?: string;
};

type SessionMemoryState = {
  summary: {
    deterministic_fact_count: number;
    superseded_fact_count: number;
    candidate_signal_count: number;
    context_line_count: number;
    memory_signal_count: number;
  };
  deterministic_facts: MemoryStateFact[];
  superseded_facts: MemoryStateFact[];
  candidate_signals: Array<{ text: string; reason: string }>;
  context_preview: string[];
  recent_memory_signals: MemorySignal[];
};

type Message = { id: number; role: 'user' | 'assistant'; content: string };

type ActivityItem = {
  id: number;
  label: string;
  detail?: string;
  expanded: boolean;
};

type SessionHistoryItem = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
};

let _uid = 0;
const uid = () => ++_uid;

const fmtBytes = (b: number) => {
  if (!b) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB'];
  let v = b,
    i = 0;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${u[i]}`;
};

// Strip <THOUGHT>...</THOUGHT> spans from the streaming token feed.
// The adapter wraps thinking-model reasoning in these markers so it doesn't
// pollute the visible chat. State (`inThought`) carries across token events
// because a span can begin and end in different chunks.
function partitionThought(
  token: string,
  inThought: boolean,
): { visible: string; thought: string; inThought: boolean } {
  const OPEN = '<THOUGHT>';
  const CLOSE = '</THOUGHT>';
  let visible = '';
  let thought = '';
  let i = 0;
  while (i < token.length) {
    if (inThought) {
      const idx = token.indexOf(CLOSE, i);
      if (idx === -1) {
        thought += token.slice(i);
        i = token.length;
      } else {
        thought += token.slice(i, idx);
        i = idx + CLOSE.length;
        inThought = false;
      }
    } else {
      const idx = token.indexOf(OPEN, i);
      if (idx === -1) {
        visible += token.slice(i);
        i = token.length;
      } else {
        visible += token.slice(i, idx);
        i = idx + OPEN.length;
        inThought = true;
      }
    }
  }
  return { visible, thought, inThought };
}

// ─── App ─────────────────────────────────────────────────────────────────────

export default function App() {
  const reduced = useReducedMotion();

  // API
  const [models, setModels] = useState<ModelsMap>({});
  const [model, setModel] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [storage, setStorage] = useState<StorageStatus | null>(null);
  const [memoryState, setMemoryState] = useState<SessionMemoryState | null>(
    null,
  );
  const [memoryLoading, setMemoryLoading] = useState(false);

  // Layout: centered = fresh start, chat = has history
  const [layout, setLayout] = useState<LayoutMode>('centered');

  // Agent state
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [phase, setPhase] = useState<Phase>('idle');
  const [activityShort, setActivityShort] = useState('');
  const [activities, setActivities] = useState<ActivityItem[]>([]);
  const [showMore, setShowMore] = useState(false);

  // Input
  const [text, setText] = useState('');
  const [attachments, setAttachments] = useState<File[]>([]);
  const [focused, setFocused] = useState(false);

  // Options
  const [optionsOpen, setOptionsOpen] = useState(false);
  const [resetConfirm, setResetConfirm] = useState(false);

  // Plan / verification / tension from richer SSE events
  const [, setStrategy] = useState('');
  const [, setVerificationIssues] = useState<string[]>([]);

  // Session history sidebar
  const [historyOpen, setHistoryOpen] = useState(false);
  const [sessionList, setSessionList] = useState<SessionHistoryItem[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const threadRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const expandTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const modelOptions = useMemo(() => {
    const opts: string[] = [];
    Object.entries(models).forEach(([p, ms]) =>
      ms.forEach((m) => opts.push(`${p}:${m}`)),
    );
    return opts;
  }, [models]);

  // Bootstrap
  useEffect(() => {
    void (async () => {
      const [mr, or, sr] = await Promise.all([
        fetch('/api/consumer/models')
          .then((r) => r.json())
          .catch(() => ({ models: {} })),
        fetch('/api/consumer/options')
          .then((r) => r.json())
          .catch(() => ({ model: '' })),
        fetch('/api/consumer/storage/status')
          .then((r) => r.json())
          .catch(() => null),
      ]);
      setModels(mr.models || {});
      if (or.model) setModel(or.model);
      setStorage(sr);
    })();
  }, []);

  // Auto-scroll thread
  useEffect(() => {
    if (threadRef.current) {
      threadRef.current.scrollTop = threadRef.current.scrollHeight;
    }
  }, [messages, streaming]);

  // Textarea auto-height
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
  }, [text]);

  // Show more activities after 5s
  useEffect(() => {
    if (isRunning) {
      expandTimerRef.current = setTimeout(() => setShowMore(true), 5000);
    } else {
      if (expandTimerRef.current) clearTimeout(expandTimerRef.current);
      setShowMore(false);
    }
    return () => {
      if (expandTimerRef.current) clearTimeout(expandTimerRef.current);
    };
  }, [isRunning]);

  const onSubmit = useCallback(
    async (e?: FormEvent) => {
      e?.preventDefault();
      if ((!text.trim() && attachments.length === 0) || isRunning) return;

      const userText = text.trim();
      const userFiles = [...attachments];

      setText('');
      setAttachments([]);
      setStreaming('');
      setActivities([]);
      setActivityShort('Thinking…');
      setStrategy('');
      setVerificationIssues([]);
      setIsRunning(true);
      setPhase('thinking');
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: 'user',
          content: userText || `[${userFiles.length} file(s)]`,
        },
      ]);

      const ab = new AbortController();
      abortRef.current = ab;

      try {
        const fd = appendOwnerId(new FormData());
        fd.append('message', userText);
        if (sessionId) fd.append('session_id', sessionId);
        if (model) fd.append('model', model);
        userFiles.forEach((f) => fd.append('files', f));

        const res = await fetch('/api/consumer/chat/stream', {
          method: 'POST',
          body: fd,
          signal: ab.signal,
        });
        const reader = res.body?.getReader();
        if (!reader) throw new Error('no stream');

        const dec = new TextDecoder();
        let buf = '';
        let acc = '';
        let thoughtAcc = '';
        let inThought = false;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const lines = buf.split('\n');
          buf = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            let ev: StreamEvent;
            try {
              ev = JSON.parse(line.slice(6));
            } catch {
              continue;
            }

            if (ev.type === 'session') setSessionId(ev.session_id);
            if (ev.type === 'phase') {
              setPhase(ev.phase);
              setActivityShort(
                {
                  thinking: 'Thinking…',
                  working: 'Working…',
                  finalizing: 'Finalizing…',
                }[ev.phase],
              );
            }
            if (ev.type === 'activity_short') setActivityShort(ev.text);
            if (ev.type === 'activity_detail') {
              setActivities((prev) =>
                [
                  {
                    id: uid(),
                    label: ev.text,
                    detail: ev.detail,
                    expanded: false,
                  },
                  ...prev,
                ].slice(0, 20),
              );
            }
            if (ev.type === 'token') {
              const part = partitionThought(ev.content, inThought);
              inThought = part.inThought;
              if (part.thought) thoughtAcc += part.thought;
              if (part.visible) {
                acc += part.visible;
                setStreaming(acc);
              }
            }
            if (ev.type === 'plan') {
              setStrategy(ev.strategy);
              setActivities((prev) =>
                [
                  {
                    id: uid(),
                    label: `Strategy: ${ev.strategy}`,
                    detail: ev.tools.join(', '),
                    expanded: false,
                  },
                  ...prev,
                ].slice(0, 20),
              );
            }
            if (ev.type === 'redraft') {
              acc = '';
              thoughtAcc = '';
              inThought = false;
              setStreaming('');
              setActivities((prev) =>
                [
                  {
                    id: uid(),
                    label: '↻ Redrafting (verification failed)',
                    detail: (ev.issues || []).join('\n'),
                    expanded: false,
                  },
                  ...prev,
                ].slice(0, 20),
              );
            }
            if (ev.type === 'verification_warning') {
              setVerificationIssues(ev.issues);
              setActivities((prev) =>
                [
                  {
                    id: uid(),
                    label: `⚠ Verification: ${ev.issues[0] || 'issue detected'}`,
                    expanded: false,
                  },
                  ...prev,
                ].slice(0, 20),
              );
            }
            if (ev.type === 'tension') {
              setActivities((prev) =>
                [
                  {
                    id: uid(),
                    label: `Tension: ${ev.summary}`,
                    expanded: false,
                  },
                  ...prev,
                ].slice(0, 20),
              );
            }
            if (ev.type === 'error') {
              setActivities((prev) => [
                { id: uid(), label: `Error: ${ev.message}`, expanded: false },
                ...prev,
              ]);
              setIsRunning(false);
              setPhase('idle');
            }
            if (ev.type === 'done') {
              if (acc) {
                setMessages((prev) => [
                  ...prev,
                  { id: uid(), role: 'assistant', content: acc },
                ]);
                setStreaming('');
              }
              if (thoughtAcc.trim()) {
                // Surface reasoning in the activity panel rather than the chat
                // bubble — keeps the conversation clean while preserving
                // observability of the model's internal channel.
                setActivities((prev) =>
                  [
                    {
                      id: uid(),
                      label: 'Reasoning',
                      detail: thoughtAcc,
                      expanded: false,
                    },
                    ...prev,
                  ].slice(0, 20),
                );
              }
              setIsRunning(false);
              setPhase('idle');
              setActivityShort('');
              setLayout('chat'); // ← switch to normal chat layout after first response
            }
          }
        }
      } catch (err: unknown) {
        if (err instanceof Error && err.name !== 'AbortError') {
          setActivities((prev) => [
            { id: uid(), label: 'Connection error', expanded: false },
            ...prev,
          ]);
        }
        setIsRunning(false);
        setPhase('idle');
      }
    },
    [text, attachments, isRunning, sessionId, model],
  );

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        void onSubmit();
      }
    },
    [onSubmit],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setIsRunning(false);
    setPhase('idle');
    setStreaming('');
  }, []);

  const saveModel = useCallback(async (m: string) => {
    setModel(m);
    await fetch('/api/consumer/options', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: m }),
    });
  }, []);

  const refreshStorage = useCallback(async () => {
    setStorage(
      await fetch('/api/consumer/storage/status').then((r) => r.json()),
    );
  }, []);

  const refreshMemoryState = useCallback(
    async (targetSessionId?: string | null) => {
      const sid = targetSessionId ?? sessionId;
      if (!sid) {
        setMemoryState(null);
        return;
      }
      setMemoryLoading(true);
      try {
        const ownerId = getOwnerId();
        const url = `/api/consumer/sessions/${sid}/memory-state?owner_id=${encodeURIComponent(ownerId)}`;
        const data = await fetch(url).then((r) => r.json());
        setMemoryState(data);
      } catch {
        setMemoryState(null);
      } finally {
        setMemoryLoading(false);
      }
    },
    [sessionId],
  );

  const backup = useCallback(async () => {
    await fetch('/api/consumer/storage/backup', { method: 'POST' });
    await refreshStorage();
  }, [refreshStorage]);

  const loadSessions = useCallback(async () => {
    setSessionsLoading(true);
    try {
      const ownerId = getOwnerId();
      const data = await fetch(
        `/api/consumer/sessions?owner_id=${encodeURIComponent(ownerId)}`,
      ).then((r) => r.json());
      setSessionList(data.sessions || []);
    } catch {
      setSessionList([]);
    } finally {
      setSessionsLoading(false);
    }
  }, []);

  const loadSession = useCallback(async (sid: string) => {
    try {
      const ownerId = getOwnerId();
      const data = await fetch(
        `/api/consumer/sessions/${sid}/messages?owner_id=${encodeURIComponent(ownerId)}`,
      ).then((r) => r.json());
      const msgs: Message[] = (data.messages || []).map(
        (m: { role: string; content: string }) => ({
          id: uid(),
          role: m.role as 'user' | 'assistant',
          content: m.content,
        }),
      );
      setSessionId(sid);
      setMessages(msgs);
      setStreaming('');
      setLayout('chat');
      setHistoryOpen(false);
    } catch {
      /* ignore */
    }
  }, []);

  const resetAll = useCallback(async () => {
    await fetch('/api/consumer/storage/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        consumer_db: true,
        consumer_sessions: true,
        backup_first: true,
      }),
    });
    setSessionId(null);
    setMessages([]);
    setStreaming('');
    setLayout('centered');
    setMemoryState(null);
    setResetConfirm(false);
    await refreshStorage();
  }, [refreshStorage]);

  useEffect(() => {
    if (optionsOpen && sessionId) {
      void refreshMemoryState(sessionId);
    }
  }, [optionsOpen, sessionId, messages.length, refreshMemoryState]);

  const toggleActivity = useCallback((id: number) => {
    setActivities((prev) =>
      prev.map((a) => (a.id === id ? { ...a, expanded: !a.expanded } : a)),
    );
  }, []);

  // Activity chips — reused in both layouts
  const ActivityChips = useCallback(
    () => (
      <div className='activities'>
        {activities.slice(0, showMore ? 10 : 3).map((act) => (
          <motion.div
            key={act.id}
            className='act-chip'
            initial={{ opacity: 0, y: 6, scale: 0.94 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={
              reduced
                ? { duration: 0 }
                : { type: 'spring', stiffness: 400, damping: 36 }
            }
            layout
          >
            <button
              className='act-chip-head'
              onClick={() => act.detail && toggleActivity(act.id)}
              style={{ cursor: act.detail ? 'pointer' : 'default' }}
            >
              <span className='act-dot' />
              <span className='act-label'>{act.label}</span>
              {act.detail && (
                <span className='act-toggle'>{act.expanded ? '−' : '+'}</span>
              )}
            </button>
            <AnimatePresence>
              {act.expanded && act.detail && (
                <motion.div
                  className='act-detail'
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  {act.detail}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        ))}
      </div>
    ),
    [activities, showMore, reduced, toggleActivity],
  );

  const spring = { type: 'spring', stiffness: 380, damping: 38 } as const;
  const ease32 = {
    type: 'tween',
    ease: [0.25, 0.1, 0.25, 1],
    duration: 0.32,
  } as const;

  return (
    <div className={`shell ${layout}`}>
      <div className='ambient' aria-hidden />

      {/* ── Top bar — chat mode only ── */}
      <AnimatePresence>
        {layout === 'chat' && (
          <motion.header
            className='topbar'
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={ease32}
          >
            <div className='brand'>
              <span className='brand-mark'>◈</span>
              SHOVS
            </div>
            <div className='topbar-right'>
              <AnimatePresence>
                {isRunning && (
                  <motion.button
                    className='cancel-btn'
                    onClick={cancel}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 10 }}
                    transition={{ duration: 0.15 }}
                  >
                    Cancel
                  </motion.button>
                )}
              </AnimatePresence>
              <button
                className='icon-btn'
                onClick={() => {
                  setHistoryOpen((v) => !v);
                  if (!historyOpen) void loadSessions();
                }}
                aria-label='Session history'
                title='Session history'
              >
                <svg
                  width='15'
                  height='15'
                  viewBox='0 0 16 16'
                  fill='none'
                  stroke='currentColor'
                  strokeWidth='1.4'
                  strokeLinecap='round'
                  strokeLinejoin='round'
                >
                  <circle cx='8' cy='8' r='6.5' />
                  <polyline points='8,4 8,8 11,10' />
                </svg>
              </button>
              <button
                className={`icon-btn ${optionsOpen ? 'active' : ''}`}
                onClick={() => setOptionsOpen((v) => !v)}
                aria-label='Options'
              >
                <svg
                  width='15'
                  height='15'
                  viewBox='0 0 15 15'
                  fill='currentColor'
                >
                  <rect x='2' y='2.5' width='11' height='1.4' rx='0.7' />
                  <rect x='2' y='6.8' width='11' height='1.4' rx='0.7' />
                  <rect x='2' y='11.1' width='11' height='1.4' rx='0.7' />
                </svg>
              </button>
            </div>
          </motion.header>
        )}
      </AnimatePresence>

      {/* ══════════════════════════════════════════════════════════════
          CENTERED LAYOUT — fresh state, input is the hero
      ══════════════════════════════════════════════════════════════ */}
      <AnimatePresence>
        {layout === 'centered' && (
          <motion.div
            className='center-stage'
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, y: -24, scale: 0.97 }}
            transition={reduced ? { duration: 0 } : ease32}
          >
            {/* Orb + status above input */}
            <div className='center-hero'>
              <motion.div
                className='center-orb'
                animate={isRunning ? { scale: [1, 1.05, 1] } : { scale: 1 }}
                transition={isRunning ? { repeat: Infinity, duration: 2 } : {}}
              >
                <AgentOrb phase={phase} size={64} />
              </motion.div>
              <AnimatePresence mode='wait'>
                <motion.h1
                  key={isRunning ? activityShort : 'idle'}
                  className='center-title'
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.22 }}
                >
                  {isRunning
                    ? activityShort || 'Working…'
                    : 'What can I help with?'}
                </motion.h1>
              </AnimatePresence>
              {isRunning && (
                <motion.p
                  className='center-phase'
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  {phase}
                </motion.p>
              )}
            </div>

            {/* Activity chips — visible during working state */}
            <AnimatePresence>
              {isRunning && activities.length > 0 && (
                <motion.div
                  className='center-activities'
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={ease32}
                >
                  <ActivityChips />
                </motion.div>
              )}
            </AnimatePresence>

            {/* Input */}
            <div className='center-input-wrap'>
              <ComposerForm
                text={text}
                setText={setText}
                attachments={attachments}
                setAttachments={setAttachments}
                focused={focused}
                setFocused={setFocused}
                isRunning={isRunning}
                onSubmit={onSubmit}
                onKeyDown={onKeyDown}
                textareaRef={textareaRef}
              />
              <motion.button
                className='center-options-btn'
                onClick={() => setOptionsOpen((v) => !v)}
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.6 }}
                whileHover={{ opacity: 1 }}
              >
                <svg
                  width='12'
                  height='12'
                  viewBox='0 0 12 12'
                  fill='currentColor'
                >
                  <rect x='1' y='1' width='10' height='1.2' rx='0.6' />
                  <rect x='1' y='5.4' width='10' height='1.2' rx='0.6' />
                  <rect x='1' y='9.8' width='10' height='1.2' rx='0.6' />
                </svg>
                Options
              </motion.button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ══════════════════════════════════════════════════════════════
          CHAT LAYOUT — after first response
      ══════════════════════════════════════════════════════════════ */}
      <AnimatePresence>
        {layout === 'chat' && (
          <motion.div
            className='chat-stage'
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
          >
            {/* Thread */}
            <div className='thread' ref={threadRef}>
              <div className='thread-inner'>
                {messages.map((msg) => (
                  <motion.div
                    key={msg.id}
                    className={`msg msg-${msg.role}`}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={reduced ? { duration: 0 } : spring}
                    layout
                  >
                    {msg.role === 'assistant' && (
                      <div className='msg-avatar'>
                        <AgentOrb phase='idle' size={22} />
                      </div>
                    )}
                    <div className='msg-bubble'>
                      {msg.role === 'assistant' ? (
                        <MarkdownMessage content={msg.content} />
                      ) : (
                        <span className='msg-text'>{msg.content}</span>
                      )}
                    </div>
                  </motion.div>
                ))}

                {/* Streaming bubble */}
                <AnimatePresence>
                  {streaming && (
                    <motion.div
                      className='msg msg-assistant'
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                    >
                      <div className='msg-avatar'>
                        <AgentOrb phase={phase} size={22} />
                      </div>
                      <div className='msg-bubble streaming'>
                        <MarkdownMessage content={streaming} />
                        <span className='cursor' aria-hidden />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Activity chips inline in chat */}
                <AnimatePresence>
                  {isRunning && (
                    <motion.div
                      className='chat-activities-wrap'
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                    >
                      <div className='chat-status-row'>
                        <AgentOrb phase={phase} size={16} />
                        <AnimatePresence mode='wait'>
                          <motion.span
                            key={activityShort}
                            className='chat-status-text'
                            initial={{ opacity: 0, x: 4 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -4 }}
                            transition={{ duration: 0.15 }}
                          >
                            {activityShort || 'Working…'}
                          </motion.span>
                        </AnimatePresence>
                      </div>
                      {activities.length > 0 && <ActivityChips />}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>

            {/* Chat input — pinned to bottom */}
            <div className='chat-input-zone'>
              <ComposerForm
                text={text}
                setText={setText}
                attachments={attachments}
                setAttachments={setAttachments}
                focused={focused}
                setFocused={setFocused}
                isRunning={isRunning}
                onSubmit={onSubmit}
                onKeyDown={onKeyDown}
                textareaRef={textareaRef}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Session history sidebar ── */}
      <AnimatePresence>
        {historyOpen && (
          <>
            <motion.div
              className='options-scrim'
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={() => setHistoryOpen(false)}
            />
            <motion.aside
              className='sidebar sidebar-left'
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={reduced ? { duration: 0 } : spring}
            >
              <div className='opt-head'>
                <span className='opt-title'>Sessions</span>
                <button
                  className='icon-btn'
                  onClick={() => setHistoryOpen(false)}
                  aria-label='Close'
                >
                  <svg
                    width='13'
                    height='13'
                    viewBox='0 0 13 13'
                    fill='none'
                    stroke='currentColor'
                    strokeWidth='1.8'
                    strokeLinecap='round'
                  >
                    <line x1='2' y1='2' x2='11' y2='11' />
                    <line x1='11' y1='2' x2='2' y2='11' />
                  </svg>
                </button>
              </div>
              <div className='opt-body'>
                <button
                  className='opt-btn'
                  style={{ width: '100%', marginBottom: 12 }}
                  onClick={() => {
                    setSessionId(null);
                    setMessages([]);
                    setStreaming('');
                    setLayout('centered');
                    setHistoryOpen(false);
                  }}
                >
                  + New Chat
                </button>
                {sessionsLoading ? (
                  <span className='opt-muted'>Loading…</span>
                ) : sessionList.length === 0 ? (
                  <span className='opt-muted'>No previous sessions.</span>
                ) : (
                  <div className='session-list'>
                    {sessionList.map((s) => (
                      <button
                        key={s.id}
                        className={`session-item ${s.id === sessionId ? 'active' : ''}`}
                        onClick={() => void loadSession(s.id)}
                      >
                        <span className='session-item-title'>
                          {s.title || 'Untitled'}
                        </span>
                        <span className='session-item-meta'>
                          {s.message_count} msg ·{' '}
                          {new Date(s.updated_at).toLocaleDateString()}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* ── Options panel ── */}
      <AnimatePresence>
        {optionsOpen && (
          <>
            <motion.div
              className='options-scrim'
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={() => setOptionsOpen(false)}
            />
            <motion.aside
              className='options-panel'
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={reduced ? { duration: 0 } : spring}
            >
              <div className='opt-head'>
                <span className='opt-title'>Configuration</span>
                <button
                  className='icon-btn'
                  onClick={() => setOptionsOpen(false)}
                  aria-label='Close'
                >
                  <svg
                    width='13'
                    height='13'
                    viewBox='0 0 13 13'
                    fill='none'
                    stroke='currentColor'
                    strokeWidth='1.8'
                    strokeLinecap='round'
                  >
                    <line x1='2' y1='2' x2='11' y2='11' />
                    <line x1='11' y1='2' x2='2' y2='11' />
                  </svg>
                </button>
              </div>
              <div className='opt-body'>
                <section className='opt-sec'>
                  <label className='opt-lbl'>Model</label>
                  <div className='sel-wrap'>
                    <select
                      className='opt-sel'
                      value={model}
                      onChange={(e) => void saveModel(e.target.value)}
                      disabled={isRunning}
                    >
                      {modelOptions.length === 0 && (
                        <option value=''>No models found</option>
                      )}
                      {modelOptions.map((o) => (
                        <option key={o} value={o}>
                          {o}
                        </option>
                      ))}
                    </select>
                    <span className='sel-caret' aria-hidden>
                      ▾
                    </span>
                  </div>
                </section>

                <section className='opt-sec'>
                  <label className='opt-lbl'>Session</label>
                  <div className='session-row'>
                    <span className='session-id'>{sessionId ?? '—'}</span>
                    {sessionId && (
                      <button
                        className='text-link'
                        onClick={() => {
                          setSessionId(null);
                          setMessages([]);
                          setStreaming('');
                          setLayout('centered');
                        }}
                      >
                        Clear
                      </button>
                    )}
                  </div>
                </section>

                <section className='opt-sec'>
                  <label className='opt-lbl'>Memory State</label>
                  {!sessionId ? (
                    <span className='opt-muted'>
                      Start a chat to inspect memory.
                    </span>
                  ) : memoryLoading ? (
                    <span className='opt-muted'>Loading memory state…</span>
                  ) : !memoryState ? (
                    <span className='opt-muted'>Memory state unavailable.</span>
                  ) : (
                    <>
                      <div className='memory-summary-grid'>
                        <div className='memory-stat-card'>
                          <span className='memory-stat-label'>Trusted</span>
                          <strong>
                            {memoryState.summary.deterministic_fact_count}
                          </strong>
                        </div>
                        <div className='memory-stat-card'>
                          <span className='memory-stat-label'>Superseded</span>
                          <strong>
                            {memoryState.summary.superseded_fact_count}
                          </strong>
                        </div>
                        <div className='memory-stat-card'>
                          <span className='memory-stat-label'>Candidates</span>
                          <strong>
                            {memoryState.summary.candidate_signal_count}
                          </strong>
                        </div>
                        <div className='memory-stat-card'>
                          <span className='memory-stat-label'>Context</span>
                          <strong>
                            {memoryState.summary.context_line_count}
                          </strong>
                        </div>
                      </div>

                      <div className='memory-panel-card'>
                        <div className='memory-panel-title'>Trusted now</div>
                        {memoryState.deterministic_facts.length === 0 ? (
                          <div className='memory-panel-empty'>
                            No trusted facts yet.
                          </div>
                        ) : (
                          memoryState.deterministic_facts
                            .slice(0, 4)
                            .map((fact, index) => (
                              <div
                                key={`${fact.subject}-${fact.predicate}-${fact.object}-${index}`}
                                className='memory-row'
                              >
                                <span>{fact.subject}</span>
                                <span>{fact.predicate}</span>
                                <span>{fact.object}</span>
                              </div>
                            ))
                        )}
                      </div>

                      {memoryState.superseded_facts.length > 0 && (
                        <div className='memory-panel-card'>
                          <div className='memory-panel-title'>Superseded</div>
                          {memoryState.superseded_facts
                            .slice(0, 3)
                            .map((fact, index) => (
                              <div
                                key={`${fact.subject}-${fact.predicate}-${fact.object}-${index}`}
                                className='memory-row subdued'
                              >
                                <span>{fact.subject}</span>
                                <span>{fact.predicate}</span>
                                <span>{fact.object}</span>
                              </div>
                            ))}
                        </div>
                      )}

                      {memoryState.candidate_signals.length > 0 && (
                        <div className='memory-panel-card'>
                          <div className='memory-panel-title'>Under review</div>
                          {memoryState.candidate_signals
                            .slice(0, 3)
                            .map((signal, index) => (
                              <div
                                key={`${signal.text}-${index}`}
                                className='memory-note'
                              >
                                {signal.text}
                                <span>{signal.reason}</span>
                              </div>
                            ))}
                        </div>
                      )}

                      {memoryState.recent_memory_signals.length > 0 && (
                        <div className='memory-panel-card'>
                          <div className='memory-panel-title'>
                            Why memory changed
                          </div>
                          {memoryState.recent_memory_signals
                            .slice(0, 3)
                            .map((signal, index) => (
                              <div
                                key={`${signal.event_type}-${index}`}
                                className='memory-note'
                              >
                                {signal.label}
                                <span>{signal.summary}</span>
                              </div>
                            ))}
                        </div>
                      )}

                      <div className='opt-acts'>
                        <button
                          className='opt-btn'
                          onClick={() => void refreshMemoryState(sessionId)}
                        >
                          Refresh
                        </button>
                      </div>
                    </>
                  )}
                </section>

                <section className='opt-sec'>
                  <label className='opt-lbl'>Storage</label>
                  <div className='stor-list'>
                    {storage?.stores ? (
                      Object.entries(storage.stores).map(([k, v]) => (
                        <div key={k} className='stor-row'>
                          <span className='stor-key'>{k}</span>
                          <span
                            className={`stor-val ${v.exists ? '' : 'miss'}`}
                          >
                            {v.exists ? fmtBytes(v.size_bytes) : 'missing'}
                          </span>
                        </div>
                      ))
                    ) : (
                      <span className='opt-muted'>Unavailable</span>
                    )}
                  </div>
                  <div className='opt-acts'>
                    <button className='opt-btn' onClick={() => void backup()}>
                      Backup
                    </button>
                    <button
                      className='opt-btn danger'
                      onClick={() => setResetConfirm(true)}
                    >
                      Reset
                    </button>
                  </div>
                  <AnimatePresence>
                    {resetConfirm && (
                      <motion.div
                        className='confirm-box'
                        initial={{ opacity: 0, y: -6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.18 }}
                      >
                        <p>Wipe consumer DB + all sessions?</p>
                        <div className='opt-acts'>
                          <button
                            className='opt-btn danger'
                            onClick={() => void resetAll()}
                          >
                            Confirm
                          </button>
                          <button
                            className='opt-btn'
                            onClick={() => setResetConfirm(false)}
                          >
                            Cancel
                          </button>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </section>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Composer form — shared between centered + chat layouts ──────────────────

interface ComposerProps {
  text: string;
  setText: (v: string) => void;
  attachments: File[];
  setAttachments: React.Dispatch<React.SetStateAction<File[]>>;
  focused: boolean;
  setFocused: (v: boolean) => void;
  isRunning: boolean;
  onSubmit: (e?: FormEvent) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
}

function ComposerForm({
  text,
  setText,
  attachments,
  setAttachments,
  focused,
  setFocused,
  isRunning,
  onSubmit,
  onKeyDown,
  textareaRef,
}: ComposerProps) {
  return (
    <form
      className={`composer ${focused ? 'focused' : ''} ${isRunning ? 'running' : ''}`}
      onSubmit={onSubmit}
    >
      {/* Attachment chips */}
      <AnimatePresence>
        {attachments.length > 0 && (
          <motion.div
            className='attach-row'
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.18 }}
          >
            {attachments.map((file) => (
              <motion.span
                key={file.name}
                className='file-chip'
                initial={{ opacity: 0, scale: 0.88 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.88 }}
                layout
              >
                <span className='file-chip-name'>
                  {file.name.length > 22
                    ? file.name.slice(0, 20) + '…'
                    : file.name}
                </span>
                <button
                  type='button'
                  className='file-chip-x'
                  onClick={() =>
                    setAttachments((p) => p.filter((f) => f.name !== file.name))
                  }
                  aria-label={`Remove ${file.name}`}
                >
                  ×
                </button>
              </motion.span>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      <div className='composer-row'>
        {/* Attach */}
        <label className='attach-btn' title='Attach' aria-label='Attach file'>
          <svg
            width='15'
            height='15'
            viewBox='0 0 16 16'
            fill='none'
            stroke='currentColor'
            strokeWidth='1.5'
            strokeLinecap='round'
            strokeLinejoin='round'
          >
            <path d='M13.5 7.5L7 14C5.3 15.7 2.6 15.7 0.9 14C-0.7 12.3-0.7 9.6 0.9 7.9L7.4 1.4C8.5.3 10.3.3 11.4 1.4 12.5 2.5 12.5 4.3 11.4 5.4L5 11.8C4.4 12.4 3.5 12.4 2.9 11.8 2.3 11.2 2.3 10.3 2.9 9.7L8.5 4.1' />
          </svg>
          <input
            type='file'
            multiple
            accept='image/*,.pdf,.txt,.md,.csv,.json,.py,.js,.ts'
            onChange={(e) =>
              setAttachments((prev) => {
                const inc = Array.from(e.target.files || []);
                const names = new Set(prev.map((f) => f.name));
                return [
                  ...prev,
                  ...inc.filter((f) => !names.has(f.name)),
                ].slice(0, 5);
              })
            }
            hidden
          />
        </label>

        <textarea
          ref={textareaRef}
          className='input-field'
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder={isRunning ? '' : 'Ask anything…'}
          rows={1}
          disabled={isRunning}
          aria-label='Message'
        />

        <button
          type='submit'
          className='send-btn'
          disabled={isRunning || (!text.trim() && attachments.length === 0)}
          aria-label='Send'
        >
          <AnimatePresence mode='wait'>
            {isRunning ? (
              <motion.span
                key='dots'
                className='dots'
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <span />
                <span />
                <span />
              </motion.span>
            ) : (
              <motion.svg
                key='arrow'
                width='14'
                height='14'
                viewBox='0 0 14 14'
                fill='currentColor'
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <path d='M1 13L13 7L1 1V5.5L9 7L1 8.5V13Z' />
              </motion.svg>
            )}
          </AnimatePresence>
        </button>
      </div>
    </form>
  );
}
