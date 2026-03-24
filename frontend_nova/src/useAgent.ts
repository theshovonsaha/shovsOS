import { useState, useEffect, useRef, useCallback } from 'react';
import { appendOwnerId, getOwnerId, withOwnerPayload, withOwnerQuery } from './owner';

const TOOL_ARG_PRIORITY = ['query', 'url', 'path', 'title', 'filename', 'language', 'command', 'prompt'];
const CONTENT_SIZE_SUMMARY_KEYS = ['code', 'html', 'content', 'svg', 'script', 'css', 'markup'];

const formatToolArgumentValue = (key: string, value: unknown): string => {
    if (typeof value === 'string') {
        const normalized = value.replace(/\s+/g, ' ').trim();
        if (!normalized) return 'empty';

        if (CONTENT_SIZE_SUMMARY_KEYS.some(token => key.toLowerCase().includes(token))) {
            return `${normalized.length} chars`;
        }

        return normalized.length > 72 ? `${normalized.slice(0, 69)}…` : normalized;
    }

    if (Array.isArray(value)) {
        return `${value.length} item${value.length === 1 ? '' : 's'}`;
    }

    if (value && typeof value === 'object') {
        const fieldCount = Object.keys(value as Record<string, unknown>).length;
        return `${fieldCount} field${fieldCount === 1 ? '' : 's'}`;
    }

    return String(value);
};

const summarizeToolArguments = (args: Record<string, unknown> = {}): string => {
    const entries = Object.entries(args);
    if (!entries.length) return 'starting…';

    const sorted = [...entries].sort(([a], [b]) => {
        const aRank = TOOL_ARG_PRIORITY.indexOf(a);
        const bRank = TOOL_ARG_PRIORITY.indexOf(b);
        return (aRank === -1 ? Number.MAX_SAFE_INTEGER : aRank) - (bRank === -1 ? Number.MAX_SAFE_INTEGER : bRank);
    });

    const displayed = sorted.slice(0, 3).map(([key, value]) => `${key}: ${formatToolArgumentValue(key, value)}`);
    const remaining = sorted.length - displayed.length;
    return remaining > 0 ? `${displayed.join(' · ')} · +${remaining} more` : displayed.join(' · ');
};

export interface Session {
    id: string;
    title: string;
    model: string;
    created_at: string;
    updated_at: string;
    message_count: number;
    context_mode?: 'v1' | 'v2' | 'v3';
}

interface AgentSettingsProfile {
    id: string;
    model: string;
    embed_model?: string;
    default_use_planner?: boolean;
    default_loop_mode?: 'auto' | 'single' | 'managed';
    default_context_mode?: 'v1' | 'v2' | 'v3';
}

export interface Attachment {
    id: string;
    file: File;
    dataURL: string | null;
}

export interface MessageBlock {
    type: 'text' | 'thought' | 'plan' | 'tool_call' | 'tool_result' | 'tool_error' | 'attachment_badge' | 'compressing';
    content: string;
    tool?: string;
    id: string;
}

export interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    files?: Attachment[];
    blocks: MessageBlock[];
    _pendingStructured?: string;
}

interface SessionTraceEvent {
    id: string;
    event_type: string;
    preview?: string;
    data?: any;
}

const STREAM_SENTINEL_RE = /(?:^\s*\[Tool Execution Turn\]\s*$|^\s*confirmation_timeout\s*$)/gim;

const sanitizeVisibleText = (value: string): string => value.replace(STREAM_SENTINEL_RE, '').replace(/\n{3,}/g, '\n\n');

const looksLikeStructuredLeak = (value: string): boolean => {
    const trimmed = sanitizeVisibleText(value).trim();
    if (!trimmed) return false;
    if (trimmed.startsWith('{"tool"') || trimmed.startsWith('{"tool_calls"')) return true;
    if (trimmed === '{' || trimmed === '[') return true;
    return (
        trimmed.startsWith('{') &&
        (trimmed.includes('"tool"') || trimmed.includes('"tool_calls"') || trimmed.includes('"arguments"'))
    );
};

const isLikelyToolJsonStart = (token: string): boolean => {
    const trimmed = token.trimStart();
    return (
        trimmed.startsWith('{"tool"') ||
        trimmed.startsWith('{"tool_calls"') ||
        trimmed.startsWith("{\n\"tool\"") ||
        trimmed.startsWith('{ "tool"') ||
        trimmed === '{'
    );
};

const INTERNAL_PRE_TOOL_CHATTER_RE = /(?:i(?:'| a)?m sorry,? but i can'?t assist with that\.?|i already have an execution plan.*|the results? .*evidence packet.*|what specific information are you looking for.*|please let me know what specific information.*|sure,? i can research .* for you\..*|<system_evidence_packet>[\s\S]*?<\/system_evidence_packet>|<system_observation>[\s\S]*?<\/system_observation>|<tool_call>[\s\S]*?<\/tool_call>|<\/?arg_key>|<\/?arg_value>)/i;

const stripInternalPreToolChatter = (value: string): string => {
    const normalized = sanitizeVisibleText(value).trim();
    if (!normalized) return '';
    return INTERNAL_PRE_TOOL_CHATTER_RE.test(normalized) ? '' : normalized;
};

const summarizeReloadedToolCall = (event: SessionTraceEvent): string => {
    const data = event.data || {};
    return data.arguments_summary || event.preview || 'tool call';
};

const summarizeReloadedToolResult = (event: SessionTraceEvent): string => {
    const data = event.data || {};
    return data.content_preview || event.preview || 'tool result';
};

const resolveModelSelection = (
    current: string,
    groupedModels: Record<string, string[]>,
): string => {
    if (!current) return current;
    if (current.includes(':')) return current;
    for (const [provider, models] of Object.entries(groupedModels || {})) {
        if (Array.isArray(models) && models.includes(current)) {
            return `${provider}:${current}`;
        }
    }
    return current;
};

const buildAssistantBlockGroupsFromTrace = (events: SessionTraceEvent[]): MessageBlock[][] => {
    const chronological = [...events].reverse();
    const groups: MessageBlock[][] = [];
    let current: MessageBlock[] = [];
    let blockIndex = 0;

    const pushBlock = (block: Omit<MessageBlock, 'id'>) => {
        current.push({ ...block, id: `trace-${blockIndex++}` });
    };

    for (const event of chronological) {
        const data = event.data || {};
        if (event.event_type === 'plan') {
            const strategy = String(data.strategy || event.preview || '').trim();
            if (strategy) pushBlock({ type: 'plan', content: strategy });
            continue;
        }
        if (event.event_type === 'tool_call') {
            pushBlock({
                type: 'tool_call',
                tool: data.tool_name || data.tool || 'tool',
                content: summarizeReloadedToolCall(event),
            });
            continue;
        }
        if (event.event_type === 'tool_result') {
            pushBlock({
                type: data.success === false ? 'tool_error' : 'tool_result',
                tool: data.tool_name || data.tool || 'tool',
                content: summarizeReloadedToolResult(event),
            });
            continue;
        }
        if (event.event_type === 'verification_warning') {
            const issues = Array.isArray(data.issues) ? data.issues.join('; ') : event.preview || 'verification warning';
            pushBlock({ type: 'plan', content: `Verification warning: ${issues}` });
            continue;
        }
        if (event.event_type === 'assistant_response') {
            groups.push(current);
            current = [];
        }
    }

    if (current.length) groups.push(current);
    return groups;
};

export function useAgent() {
    const [health, setHealth] = useState<{ status: string; ollama: boolean }>({ status: 'connecting...', ollama: false });
    const [models, setModels] = useState<Record<string, string[]>>({ ollama: ['llama3.2'] });
    const [embedModels, setEmbedModels] = useState<Record<string, string[]>>({ 'ollama': ['nomic-embed-text'] });
    const [tools, setTools] = useState<any[]>([]);
    const [sessions, setSessions] = useState<Session[]>([]);
    const [currentSessionId, setCurrentSessionId] = useState<string | null>(localStorage.getItem('shovs_current_sid'));
    const [activeAgentId, setActiveAgentId] = useState<string | null>(localStorage.getItem('shovs_active_agent_id'));
    const [currentModel, setCurrentModel] = useState<string>(localStorage.getItem('shovs_model') || '');
    const [currentSearchBackend, setCurrentSearchBackend] = useState<string>(localStorage.getItem('shovs_search_backend') || 'auto');
    const [currentSearchEngine, setCurrentSearchEngine] = useState<string>(localStorage.getItem('shovs_search_engine') || 'auto');
    const [messages, setMessages] = useState<Message[]>([]);
    const [contextLines, setContextLines] = useState(0);
    const [contextMode, setContextMode] = useState<'v1' | 'v2' | 'v3'>('v1');
    const [isStreaming, setIsStreaming] = useState(false);
    const [pendingFiles, setPendingFiles] = useState<Attachment[]>([]);
    const [forcedTools, setForcedTools] = useState<string[]>([]);

    // V10 Layer Controls
    const [usePlanner, setUsePlanner] = useState<boolean>(localStorage.getItem('shovs_use_planner') !== 'false');
    const [loopMode, setLoopMode] = useState<'auto' | 'single' | 'managed'>(
        (localStorage.getItem('shovs_loop_mode') as 'auto' | 'single' | 'managed') || 'auto'
    );
    const [maxToolCalls, setMaxToolCalls] = useState<string>(localStorage.getItem('shovs_max_tool_calls') || '');
    const [maxTurns, setMaxTurns] = useState<string>(localStorage.getItem('shovs_max_turns') || '');
    const [plannerModel, setPlannerModel] = useState<string>(localStorage.getItem('shovs_planner_model') || '');
    const [contextModel, setContextModel] = useState<string>(localStorage.getItem('shovs_context_model') || 'deepseek-r1:8b');
    const [embedModel, setEmbedModel] = useState<string>(localStorage.getItem('shovs_embed_model') || 'ollama:nomic-embed-text');

    // Voice / Jarvis States
    const [isListening, setIsListening] = useState(false);
    const [speaking, setSpeaking] = useState(false);
    const [lastUserText, setLastUserText] = useState('');
    const [currentToken, setCurrentToken] = useState('');
    const [lastAgentResponse, setLastAgentResponse] = useState('');
    const [voiceStatus, setVoiceStatus] = useState<'idle' | 'recording' | 'processing' | 'speaking'>('idle');

    // Voice Settings
    const [voiceSensitivity, setVoiceSensitivity] = useState<number>(Number(localStorage.getItem('shovs_voice_sensitivity')) || 0.5);
    const [voiceModel, setVoiceModel] = useState<string>(localStorage.getItem('shovs_voice_model') || 'aura-orion-en');

    // Granular Agentic Visibility Controls
    const [showPlannerLog, setShowPlannerLog] = useState<boolean>(localStorage.getItem('shovs_show_planner') !== 'false');
    const [showActorThought, setShowActorThought] = useState<boolean>(localStorage.getItem('shovs_show_actor') !== 'false');
    const [showObserverActivity, setShowObserverActivity] = useState<boolean>(localStorage.getItem('shovs_show_observer') === 'true');

    const wsRef = useRef<WebSocket | null>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const ttsChunksRef = useRef<ArrayBuffer[]>([]);

    const isSendingRef = useRef(false);
    const bottomRef = useRef<HTMLDivElement>(null);
    const conversationRef = useRef<HTMLElement | null>(null);

    useEffect(() => { fetchHealth(); fetchModels(); fetchTools(); }, []);
    useEffect(() => { fetchSessions(); }, [activeAgentId]);
    useEffect(() => {
        const applyProfileDefaults = async () => {
            if (!activeAgentId) return;
            try {
                const profile: AgentSettingsProfile = await fetch(withOwnerQuery(`/api/agents/${activeAgentId}`)).then(r => r.json());
                if (profile.model) setCurrentModel(profile.model);
                if (profile.embed_model) setEmbedModel(profile.embed_model);
                setUsePlanner(profile.default_use_planner ?? true);
                setLoopMode(profile.default_loop_mode || 'auto');
                setContextMode(profile.default_context_mode || 'v2');
            } catch (e) {
                console.error('Failed to load agent defaults', e);
            }
        };
        applyProfileDefaults();
    }, [activeAgentId]);

    useEffect(() => {
        if (currentModel) localStorage.setItem('shovs_model', currentModel);
    }, [currentModel]);

    useEffect(() => {
        localStorage.setItem('shovs_search_backend', currentSearchBackend);
    }, [currentSearchBackend]);

    useEffect(() => {
        localStorage.setItem('shovs_search_engine', currentSearchEngine);
    }, [currentSearchEngine]);

    useEffect(() => {
        localStorage.setItem('shovs_use_planner', usePlanner.toString());
    }, [usePlanner]);

    useEffect(() => {
        localStorage.setItem('shovs_loop_mode', loopMode);
    }, [loopMode]);

    useEffect(() => {
        if (maxToolCalls) localStorage.setItem('shovs_max_tool_calls', maxToolCalls);
        else localStorage.removeItem('shovs_max_tool_calls');
    }, [maxToolCalls]);

    useEffect(() => {
        if (maxTurns) localStorage.setItem('shovs_max_turns', maxTurns);
        else localStorage.removeItem('shovs_max_turns');
    }, [maxTurns]);

    useEffect(() => {
        localStorage.setItem('shovs_planner_model', plannerModel);
    }, [plannerModel]);

    useEffect(() => {
        localStorage.setItem('shovs_context_model', contextModel);
    }, [contextModel]);

    useEffect(() => {
        localStorage.setItem('shovs_embed_model', embedModel);
    }, [embedModel]);

    useEffect(() => {
        localStorage.setItem('shovs_show_planner', showPlannerLog.toString());
    }, [showPlannerLog]);

    useEffect(() => {
        localStorage.setItem('shovs_show_actor', showActorThought.toString());
    }, [showActorThought]);

    useEffect(() => {
        localStorage.setItem('shovs_show_observer', showObserverActivity.toString());
    }, [showObserverActivity]);

    useEffect(() => {
        localStorage.setItem('shovs_voice_sensitivity', voiceSensitivity.toString());
    }, [voiceSensitivity]);

    useEffect(() => {
        localStorage.setItem('shovs_voice_model', voiceModel);
    }, [voiceModel]);

    useEffect(() => {
        if (activeAgentId) localStorage.setItem('shovs_active_agent_id', activeAgentId);
        else localStorage.removeItem('shovs_active_agent_id');
    }, [activeAgentId]);

    useEffect(() => {
        if (currentSessionId) localStorage.setItem('shovs_current_sid', currentSessionId);
        else localStorage.removeItem('shovs_current_sid');
    }, [currentSessionId]);

    // Auto-load session on first mount if SID exists
    useEffect(() => {
        const savedSid = localStorage.getItem('shovs_current_sid');
        if (savedSid) {
            loadSession(savedSid);
        }
    }, []);

    useEffect(() => {
        const container = conversationRef.current;
        if (!container) return;
        container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
    }, [messages, isStreaming]);

    const fetchHealth = async () => {
        try {
            const data = await fetch('/api/health').then(r => r.json());
            setHealth(data);
        } catch { setHealth({ status: 'error', ollama: false }); }
    };

    const fetchModels = async () => {
        try {
            const data = await fetch('/api/models').then(r => r.json());
            if (data.models) {
                setModels(data.models);
                const available = new Set<string>();
                const providers = Object.keys(data.models);
                for (const p of providers) {
                    const list = Array.isArray(data.models[p]) ? data.models[p] : [];
                    for (const model of list) {
                        available.add(`${p}:${model}`);
                    }
                }

                // Normalize legacy non-prefixed model values.
                const normalizedCurrent = resolveModelSelection(currentModel, data.models);

                // If current selection is missing/unavailable, pick first available.
                if (!normalizedCurrent || !available.has(normalizedCurrent)) {
                    for (const p of providers) {
                        const list = Array.isArray(data.models[p]) ? data.models[p] : [];
                        if (list.length > 0) {
                            setCurrentModel(`${p}:${list[0]}`);
                            break;
                        }
                    }
                } else if (normalizedCurrent !== currentModel) {
                    setCurrentModel(normalizedCurrent);
                }
            }
            if (data.embeddings) {
                const grouped: Record<string, string[]> = {};
                data.embeddings.forEach((e: string) => {
                    const [prov, core] = e.split(':');
                    if (!grouped[prov]) grouped[prov] = [];
                    grouped[prov].push(core);
                });
                setEmbedModels(grouped);
            }
        } catch { }
    };

    const fetchTools = async () => {
        try {
            const data = await fetch('/api/tools').then(r => r.json());
            if (data.tools?.length) setTools(data.tools);
        } catch { }
    };

    const fetchSessions = useCallback(async () => {
        try {
            const url = activeAgentId
                ? withOwnerQuery(`/api/sessions?agent_id=${encodeURIComponent(activeAgentId)}`)
                : withOwnerQuery('/api/sessions');
            const data = await fetch(url).then(r => r.json());
            setSessions(data.sessions || []);
        } catch { }
    }, [activeAgentId]);

    const loadSession = async (id: string) => {
        if (isSendingRef.current) return;
        try {
            const [data, traceData] = await Promise.all([
                fetch(withOwnerQuery(`/api/sessions/${id}`)).then(r => r.json()),
                fetch(withOwnerQuery(`/api/logs/traces/recent?session_id=${encodeURIComponent(id)}&limit=260`)).then(r => r.json()).catch(() => ({ events: [] })),
            ]);
            setCurrentSessionId(id);
            if (data.model) setCurrentModel(data.model);
            const traceBlockGroups = buildAssistantBlockGroupsFromTrace(Array.isArray(traceData?.events) ? traceData.events : []);
            let assistantIndex = 0;
            const loaded: Message[] = (data.history || []).map((m: any, i: number) => {
                let blocks: MessageBlock[] = [];
                let content = sanitizeVisibleText(m.content || '');

                if (m.role === 'assistant' && content.includes('<SYSTEM_TOOL_RESULT')) {
                    // Extract tool results explicitly
                    const parts = content.split(/<SYSTEM_TOOL_RESULT name="([^"]+)">/);
                    if (parts[0].trim()) {
                        blocks.push({ id: `b-${i}-txt`, type: 'text', content: parts[0].trim() });
                    }
                    for (let j = 1; j < parts.length; j += 2) {
                        const toolName = parts[j];
                        let toolContent = parts[j + 1] || '';
                        const closeIdx = toolContent.indexOf('</SYSTEM_TOOL_RESULT>');
                        if (closeIdx !== -1) {
                            const actualContent = toolContent.slice(0, closeIdx).trim();
                            blocks.push({
                                id: `b-${i}-t${j}`,
                                type: 'tool_result',
                                tool: toolName,
                                content: actualContent
                            });
                            const remainder = toolContent.slice(closeIdx + '</SYSTEM_TOOL_RESULT>'.length).trim();
                            if (remainder) {
                                blocks.push({ id: `b-${i}-rem${j}`, type: 'text', content: remainder });
                            }
                        }
                    }
                } else if (m.role === 'assistant' && content.includes('<think>')) {
                    // Reconstruct thought blocks
                    const parts = content.split('<think>');
                    if (parts[0].trim()) blocks.push({ id: `b-${i}-t1`, type: 'text', content: parts[0].trim() });
                    if (parts[1]) {
                        const tp = parts[1].split('</think>');
                        blocks.push({ id: `b-${i}-th`, type: 'thought', content: tp[0].trim() });
                        if (tp[1]?.trim()) blocks.push({ id: `b-${i}-t2`, type: 'text', content: tp[1].trim() });
                    }
                } else {
                    blocks = [{ id: `b-${i}`, type: 'text' as const, content }];
                }

                if (m.role === 'assistant') {
                    const traceBlocks = traceBlockGroups[assistantIndex] || [];
                    assistantIndex += 1;
                    const visibleTextBlocks = blocks.filter((block) => block.type === 'text' || block.type === 'thought');
                    const nonTextBlocks = blocks.filter((block) => block.type !== 'text' && block.type !== 'thought');
                    blocks = [...traceBlocks, ...nonTextBlocks, ...visibleTextBlocks];
                    if (!blocks.length) {
                        blocks = [{ id: `b-${i}`, type: 'text' as const, content }];
                    }
                }

                return {
                    id: `hist-${i}`,
                    role: m.role,
                    content,
                    blocks,
                };
            });
            setMessages(loaded);
            setContextLines(data.context_lines || 0);
            setContextMode(data.context_mode === 'v2' ? 'v2' : data.context_mode === 'v3' ? 'v3' : 'v1');
            fetchSessions();
        } catch (e) { console.error(e); }
    };

    const setSessionContextMode = async (mode: 'v1' | 'v2' | 'v3') => {
        try {
            let sessionId = currentSessionId;

            // Allow toggling even before first message by creating a session on demand.
            if (!sessionId) {
                const createRes = await fetch('/api/sessions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(withOwnerPayload({
                        agent_id: activeAgentId || 'default',
                        model: currentModel || undefined,
                        context_mode: mode,
                    })),
                });
                const createData = await createRes.json();
                if (!createRes.ok) {
                    throw new Error(createData?.detail || `HTTP ${createRes.status}`);
                }
                sessionId = createData.id;
                setCurrentSessionId(sessionId);
                setContextMode(createData.context_mode === 'v2' ? 'v2' : createData.context_mode === 'v3' ? 'v3' : 'v1');
                await fetchSessions();
                return;
            }

            const res = await fetch(`/api/sessions/${sessionId}/context-mode`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(withOwnerPayload({ mode })),
            });
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data?.detail || `HTTP ${res.status}`);
            }
            setContextMode(data.context_mode === 'v2' ? 'v2' : data.context_mode === 'v3' ? 'v3' : 'v1');
            await fetchSessions();
        } catch (e) {
            console.error('Failed to set context mode', e);
        }
    };

    const clearSessionContext = async () => {
        if (!currentSessionId) return;
        try {
            await fetch(withOwnerQuery(`/api/sessions/${currentSessionId}/clear_context`), { method: 'POST' });
            setContextLines(0);
        } catch (e) {
            console.error('Failed to clear context', e);
        }
    };

    const newSession = () => {
        if (isSendingRef.current) return;
        setCurrentSessionId(null);
        setMessages([]);
        setContextLines(0);
        fetchSessions();
    };

    const deleteSession = async (id: string) => {
        try {
            await fetch(withOwnerQuery(`/api/sessions/${id}`), { method: 'DELETE' });
            if (id === currentSessionId) newSession();
            else fetchSessions();
        } catch { }
    };

    const addFiles = (filesList: File[]) => {
        const newAttachments = filesList.map(file => {
            const id = Math.random().toString(36).slice(2);
            const attachment: Attachment = { id, file, dataURL: null };
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = e => setPendingFiles(prev =>
                    prev.map(p => p.id === id ? { ...p, dataURL: e.target?.result as string } : p)
                );
                reader.readAsDataURL(file);
            }
            return attachment;
        });
        setPendingFiles(prev => [...prev, ...newAttachments]);
    };

    const removeFile = (id: string) => setPendingFiles(prev => prev.filter(f => f.id !== id));

    const sendMessage = async (text: string) => {
        if (isSendingRef.current || (!text.trim() && !pendingFiles.length)) return;

        isSendingRef.current = true;
        setIsStreaming(true);

        const filesToSend = [...pendingFiles];
        setPendingFiles([]);

        const userMsgId = Date.now().toString();
        const assistantMsgId = (Date.now() + 1).toString();

        setMessages(prev => [
            ...prev,
            { id: userMsgId, role: 'user', content: text, files: filesToSend, blocks: [] },
            { id: assistantMsgId, role: 'assistant', content: '', blocks: [] },
        ]);

        try {
            const fd = appendOwnerId(new FormData());
            fd.append('message', text || '(see attached files)');
            if (currentSessionId) fd.append('session_id', currentSessionId);
            if (activeAgentId) fd.append('agent_id', activeAgentId);
            fd.append('model', currentModel);
            fd.append('search_backend', currentSearchBackend);
            fd.append('search_engine', currentSearchEngine); // PASS TO BACKEND!
            fd.append('planner_model', plannerModel);
            fd.append('context_model', contextModel);
            fd.append('context_mode', contextMode);
            fd.append('embed_model', embedModel);
            fd.append('use_planner', usePlanner.toString());
            fd.append('loop_mode', loopMode);
            if (maxToolCalls.trim()) fd.append('max_tool_calls', maxToolCalls.trim());
            if (maxTurns.trim()) fd.append('max_turns', maxTurns.trim());

            fd.append('forced_tools_json', JSON.stringify(forcedTools));
            filesToSend.forEach(f => fd.append('files', f.file));

            const res = await fetch('/api/chat/stream', { method: 'POST', body: fd });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const reader = res.body?.getReader();
            if (!reader) throw new Error('No reader');

            const decoder = new TextDecoder();
            let buf = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });

                const lines = buf.split('\n');
                buf = lines.pop() || '';

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    let ev: any;
                    try { ev = JSON.parse(line.slice(6)); } catch { continue; }

                    setMessages(prev => {
                        const next = [...prev];
                        const msg = next[next.length - 1];
                        if (msg.role !== 'assistant') return prev;

                        const mkId = () => Math.random().toString(36).slice(2);
                        const addBlock = (block: Omit<MessageBlock, 'id'>) => {
                            msg.blocks = [...msg.blocks, { ...block, id: mkId() }];
                        };

                        const lastBlock = msg.blocks[msg.blocks.length - 1];

                        const appendToVisibleBlock = (type: 'text' | 'thought', content: string) => {
                            const safeContent = sanitizeVisibleText(content);
                            if (!safeContent) return;
                            const visibleBlock = msg.blocks[msg.blocks.length - 1];
                            if (visibleBlock?.type === type) {
                                visibleBlock.content = sanitizeVisibleText(`${visibleBlock.content}${safeContent}`);
                            } else {
                                addBlock({ type, content: safeContent });
                            }
                            msg.content = sanitizeVisibleText(`${msg.content}${safeContent}`);
                        };

                        const stripTrailingStructuredNoise = () => {
                            msg._pendingStructured = '';
                            while (msg.blocks.length > 0) {
                                const trailing = msg.blocks[msg.blocks.length - 1];
                                if (trailing.type !== 'text') break;
                                const cleaned = sanitizeVisibleText(trailing.content).trimEnd();
                                if (!cleaned) {
                                    msg.blocks = msg.blocks.slice(0, -1);
                                    continue;
                                }
                                if (looksLikeStructuredLeak(cleaned)) {
                                    msg.blocks = msg.blocks.slice(0, -1);
                                    continue;
                                }
                                const withoutChatter = stripInternalPreToolChatter(cleaned);
                                if (!withoutChatter) {
                                    msg.blocks = msg.blocks.slice(0, -1);
                                    continue;
                                }
                                trailing.content = withoutChatter;
                                break;
                            }
                            msg.content = sanitizeVisibleText(
                                msg.blocks
                                    .filter(block => block.type === 'text')
                                    .map(block => block.content)
                                    .join('')
                            ).trim();
                        };

                        switch (ev.type) {
                            case 'session':
                                setCurrentSessionId(ev.session_id);
                                break;

                            case 'plan':
                                addBlock({
                                    type: 'plan',
                                    content: ev.strategy || 'Planning strategy...'
                                });
                                break;

                            case 'attachment':
                                addBlock({
                                    type: 'attachment_badge',
                                    content: ev.ok
                                        ? `✓ ${ev.filename} (${ev.file_type})`
                                        : `✗ ${ev.filename}: ${ev.error}`,
                                });
                                break;

                            case 'token':
                                {
                                const token = String(ev.content || '');
                                if (!token) break;

                                // Thinking Tag Detection Logic
                                if (token.includes('<think>')) {
                                    const parts = token.split('<think>');
                                    if (parts[0]) {
                                        appendToVisibleBlock('text', parts[0]);
                                    }
                                    addBlock({ type: 'thought', content: sanitizeVisibleText(parts[1] || '') });
                                } else if (token.includes('</think>')) {
                                    const parts = token.split('</think>');
                                    if (lastBlock?.type === 'thought') {
                                        lastBlock.content = sanitizeVisibleText(`${lastBlock.content}${parts[0]}`);
                                    }
                                    appendToVisibleBlock('text', parts[1] || '');
                                } else if (msg._pendingStructured || isLikelyToolJsonStart(token)) {
                                    msg._pendingStructured = `${msg._pendingStructured || ''}${token}`;
                                } else {
                                    const activeTextBlock = msg.blocks[msg.blocks.length - 1];
                                    if (activeTextBlock?.type === 'text' || activeTextBlock?.type === 'thought') {
                                        activeTextBlock.content = sanitizeVisibleText(`${activeTextBlock.content}${token}`);
                                        if (activeTextBlock.type === 'text') {
                                            msg.content = sanitizeVisibleText(`${msg.content}${token}`);
                                        }
                                    } else {
                                        appendToVisibleBlock('text', token);
                                    }
                                }
                                break;
                                }

                            case 'tool_call':
                                stripTrailingStructuredNoise();
                                addBlock({
                                    type: 'tool_call',
                                    tool: ev.tool_name,
                                    content: summarizeToolArguments(ev.arguments || {}),
                                });
                                break;

                            case 'tool_running':
                                // Optional: could update the last tool_call block to show spinner
                                break;

                            case 'tool_result':
                                stripTrailingStructuredNoise();
                                addBlock({
                                    type: ev.success ? 'tool_result' : 'tool_error',
                                    tool: ev.tool_name,
                                    content: ev.content || (ev.success ? 'completed' : 'failed'),
                                });
                                break;

                            case 'retract_last_tokens':
                                stripTrailingStructuredNoise();
                                break;

                            case 'compressing':
                                addBlock({ type: 'compressing', content: 'compressing context…' });
                                break;

                            case 'context_updated':
                                msg.blocks = msg.blocks.filter(b => b.type !== 'compressing');
                                setContextLines(ev.lines);
                                fetchSessions();
                                break;

                            case 'error':
                                stripTrailingStructuredNoise();
                                addBlock({ type: 'text', content: `\n\n⚠ ${sanitizeVisibleText(ev.message || 'Unknown error')}` });
                                break;
                        }

                        return next;
                    });
                }
            }

            setMessages(prev => {
                const next = [...prev];
                const msg = next[next.length - 1];
                if (msg?.role !== 'assistant' || !msg._pendingStructured) return prev;

                const leftover = sanitizeVisibleText(msg._pendingStructured).trim();
                msg._pendingStructured = '';
                if (leftover && !looksLikeStructuredLeak(leftover)) {
                    msg.blocks = [...msg.blocks, {
                        id: `flush-${Date.now()}`,
                        type: 'text',
                        content: leftover,
                    }];
                    msg.content = sanitizeVisibleText(`${msg.content}${leftover}`);
                }
                return next;
            });
        } catch (e: any) {
            setMessages(prev => {
                const next = [...prev];
                const msg = next[next.length - 1];
                if (msg.role === 'assistant') {
                    msg._pendingStructured = '';
                    msg.blocks = [...msg.blocks, {
                        id: 'err', type: 'text',
                        content: `\n\n⚠ connection error: ${sanitizeVisibleText(e.message)}`,
                    }];
                }
                return next;
            });
        } finally {
            setIsStreaming(false);
            isSendingRef.current = false;
        }
    };

    const startRecording = async () => {
        if (!activeAgentId) return;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const recorder = new MediaRecorder(stream);
            mediaRecorderRef.current = recorder;

            recorder.ondataavailable = (e) => {
                if (e.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
                    wsRef.current.send(e.data);
                }
            };

            recorder.onstop = () => {
                if (wsRef.current?.readyState === WebSocket.OPEN) {
                    wsRef.current.send(JSON.stringify({ type: 'stt_end' }));
                }
                stream.getTracks().forEach(track => track.stop());
            };

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/api/ws/voice`;
            const ws = new WebSocket(wsUrl);
            ws.binaryType = 'arraybuffer';
            wsRef.current = ws;

            ws.onopen = () => {
                ws.send(JSON.stringify({
                    type: 'config',
                    session_id: currentSessionId,
                    agent_id: activeAgentId,
                    model: currentModel,
                    owner_id: getOwnerId(),
                    voice_model: voiceModel,
                    sensitivity: voiceSensitivity
                }));
            };

            ws.onmessage = async (e) => {
                if (typeof e.data === 'string') {
                    const msg = JSON.parse(e.data);
                    switch (msg.type) {
                        case 'config_ack':
                            recorder.start(250);
                            setIsListening(true);
                            setVoiceStatus('recording');
                            setLastUserText('');
                            break;
                        case 'stt_result':
                            if (msg.text) {
                                setLastUserText(msg.text);

                                if (msg.is_final) {
                                    setVoiceStatus('processing');
                                } else {
                                    setVoiceStatus('recording');
                                }

                                // Barge-in: If we are speaking and user starts talking, stop!
                                if (speaking) {
                                    stopSpeaking();
                                }

                                if (msg.is_final) {
                                    // Mirror to chat history only when final
                                    setMessages(prev => [...prev, {
                                        id: 'u-' + Date.now(),
                                        role: 'user',
                                        content: msg.text,
                                        blocks: [{ id: 'b-' + Date.now(), type: 'text', content: msg.text }]
                                    }]);
                                    setLastAgentResponse('');
                                    setCurrentToken('');
                                }
                            }
                            break;
                        case 'agent_token':
                            setCurrentToken(prev => prev + msg.content);
                            break;
                        case 'agent_done':
                            setLastAgentResponse(msg.full_response);
                            setCurrentToken('');
                            // Mirror agent response to chat history
                            setMessages(prev => [...prev, {
                                id: 'a-' + Date.now(),
                                role: 'assistant',
                                content: msg.full_response,
                                blocks: [{ id: 'ba-' + Date.now(), type: 'text', content: msg.full_response }]
                            }]);
                            break;
                        case 'tts_start':
                            setSpeaking(true);
                            setVoiceStatus('speaking');
                            ttsChunksRef.current = [];
                            break;
                        case 'tts_end':
                            setSpeaking(false);
                            setVoiceStatus('idle');
                            if (ttsChunksRef.current.length > 0) {
                                playBufferedAudio();
                            }
                            if (wsRef.current) wsRef.current.close();
                            break;
                        case 'error':
                            console.error('Voice Error:', msg.message);
                            stopRecording();
                            setSpeaking(false);
                            setVoiceStatus('idle');
                            break;
                    }
                } else {
                    // Binary audio data (TTS)
                    if (e.data instanceof ArrayBuffer) {
                        ttsChunksRef.current.push(e.data);
                    }
                }
            };

            ws.onclose = () => {
                setIsListening(false);
                setVoiceStatus('idle');
            };

        } catch (err: any) {
            console.error('Mic error:', err);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }
        setIsListening(false);
    };

    const stopSpeaking = () => {
        if (wsRef.current) wsRef.current.close();
        setSpeaking(false);
        setVoiceStatus('idle');
    };

    const playBufferedAudio = async () => {
        if (ttsChunksRef.current.length === 0) return;

        // Merge chunks
        const totalLen = ttsChunksRef.current.reduce((acc, c) => acc + c.byteLength, 0);
        const merged = new Uint8Array(totalLen);
        let offset = 0;
        for (const chunk of ttsChunksRef.current) {
            merged.set(new Uint8Array(chunk), offset);
            offset += chunk.byteLength;
        }
        ttsChunksRef.current = [];

        if (!audioContextRef.current) {
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        }
        const ctx = audioContextRef.current;
        try {
            const buffer = await ctx.decodeAudioData(merged.buffer);
            const source = ctx.createBufferSource();
            source.buffer = buffer;
            source.connect(ctx.destination);
            source.start();
        } catch (e) {
            console.error('Failed to decode/play buffered audio:', e);
        }
    };

    // Guardrails
    const [pendingConfirmation, setPendingConfirmation] = useState<any | null>(null);

    // Guardrail stream
    useEffect(() => {
        if (!currentSessionId) return;
        const streamUrl = `/api/guardrails/stream/${currentSessionId}`;
        console.log(`[Guardrails] Connecting to SSE stream: ${streamUrl}`);
        const es = new EventSource(streamUrl);
        es.addEventListener('confirmation_required', (e: any) => {
            console.log('[Guardrails] Confirmation Required:', e.data);
            const data = JSON.parse(e.data);
            setPendingConfirmation(data);
        });
        es.onerror = (err) => console.error('[Guardrails] SSE Error:', err);
        return () => es.close();
    }, [currentSessionId]);

    const approveConfirmation = async (callId: string) => {
        console.log(`[Guardrails] Approving call: ${callId}`);
        try {
            const res = await fetch(`/api/guardrails/approve/${callId}`, { method: 'POST' });
            if (!res.ok) console.error(`[Guardrails] Approve failed: ${res.status}`);
            setPendingConfirmation(null);
        } catch (e) {
            console.error('[Guardrails] Failed to approve', e);
        }
    };

    const denyConfirmation = async (callId: string, reason = 'User denied') => {
        console.log(`[Guardrails] Denying call: ${callId} reason: ${reason}`);
        try {
            const res = await fetch(`/api/guardrails/deny/${callId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ reason }),
            });
            if (!res.ok) console.error(`[Guardrails] Deny failed: ${res.status}`);
            setPendingConfirmation(null);
        } catch (e) {
            console.error('[Guardrails] Failed to deny', e);
        }
    };

    const stopExecution = async () => {
        if (!currentSessionId) return;
        try {
            await fetch(withOwnerQuery(`/api/sessions/${currentSessionId}/stop`), { method: 'POST' });
        } catch (e) {
            console.error('Failed to stop execution', e);
        }
    };

    return {
        health, models, tools, sessions, currentSessionId,
        activeAgentId, setActiveAgentId,
        currentModel, setCurrentModel,
        currentSearchBackend, setCurrentSearchBackend,
        currentSearchEngine, setCurrentSearchEngine,
        messages, contextLines,
        contextMode, setSessionContextMode,
        isStreaming, pendingFiles,
        forcedTools, setForcedTools,
        isListening, speaking, voiceStatus,
        lastUserText, currentToken, lastAgentResponse,
        startRecording, stopRecording, stopSpeaking,
        usePlanner, setUsePlanner,
        loopMode, setLoopMode,
        maxToolCalls, setMaxToolCalls,
        maxTurns, setMaxTurns,
        plannerModel, setPlannerModel,
        contextModel, setContextModel,
        embedModel, setEmbedModel, embedModels,
        voiceSensitivity, setVoiceSensitivity,
        voiceModel,
        setVoiceModel,
        showPlannerLog,
        setShowPlannerLog,
        showActorThought,
        setShowActorThought,
        showObserverActivity,
        setShowObserverActivity,
        clearSessionContext,
        loadSession, newSession, deleteSession,
        addFiles, removeFile, sendMessage, stopExecution, bottomRef, conversationRef,
        pendingConfirmation, approveConfirmation, denyConfirmation,
    };
}
