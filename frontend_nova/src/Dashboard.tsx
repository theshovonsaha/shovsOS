import React, { useState, useEffect } from 'react';
import { PremiumSelect } from './components/PremiumSelect';
import { withOwnerPayload, withOwnerQuery } from './owner';

interface AgentProfile {
    id: string;
    name: string;
    description: string;
    model: string;
    embed_model?: string;
    system_prompt?: string;
    tools: string[];
    workspace_path?: string | null;
    bootstrap_files?: string[];
    bootstrap_max_chars?: number;
    default_use_planner?: boolean;
    default_loop_mode?: 'auto' | 'single' | 'managed';
    default_context_mode?: 'v1' | 'v2' | 'v3';
}

interface DashboardProps {
    onSelectAgent: (id: string) => void;
    embedModels: Record<string, string[]>;
}

type AgentPresetId = 'research' | 'coding' | 'operator' | 'consumer';

const AGENT_PRESETS: Record<AgentPresetId, {
    label: string;
    description: string;
    systemPrompt: string;
    tools: string[];
    defaultLoopMode: 'auto' | 'single' | 'managed';
    defaultContextMode: 'v1' | 'v2' | 'v3';
    defaultUsePlanner: boolean;
    bootstrapFiles: string[];
}> = {
    research: {
        label: 'Research',
        description: 'Evidence-first research agent for web investigation, comparison, and concise reports.',
        systemPrompt: 'Act as a rigorous researcher. Preserve exact domains and names, gather evidence before conclusions, prefer first-party sources when evaluating a product, and surface what remains unverified clearly.',
        tools: ['web_search', 'web_fetch', 'query_memory', 'store_memory'],
        defaultLoopMode: 'managed',
        defaultContextMode: 'v2',
        defaultUsePlanner: true,
        bootstrapFiles: ['AGENTS.md', 'IDENTITY.md', 'SOUL.md', 'TOOLS.md'],
    },
    coding: {
        label: 'Coding',
        description: 'Workspace-grounded coding agent optimized for implementation, reading files, and precise edits.',
        systemPrompt: 'Act as a disciplined coding operator. Work from the local workspace first, preserve project conventions, keep edits precise, and avoid speculative changes.',
        tools: ['file_view', 'file_create', 'file_str_replace', 'bash', 'query_memory', 'store_memory'],
        defaultLoopMode: 'managed',
        defaultContextMode: 'v2',
        defaultUsePlanner: true,
        bootstrapFiles: ['AGENTS.md', 'IDENTITY.md', 'SOUL.md', 'TOOLS.md'],
    },
    operator: {
        label: 'Operator',
        description: 'Low-noise operational agent for diagnostics, monitoring, and exact next-step execution.',
        systemPrompt: 'Act as an operations-focused runtime operator. Minimize chatter, prioritize exact diagnostics, explain failure modes clearly, and prefer decisive next steps over broad exploration.',
        tools: ['bash', 'file_view', 'web_search', 'query_memory', 'store_memory'],
        defaultLoopMode: 'single',
        defaultContextMode: 'v2',
        defaultUsePlanner: true,
        bootstrapFiles: ['IDENTITY.md', 'SOUL.md', 'TOOLS.md'],
    },
    consumer: {
        label: 'Consumer',
        description: 'Plain-language assistant tuned for clarity, guardrails, and minimal UI-facing noise.',
        systemPrompt: 'Act as a clear, calm assistant. Prefer plain text, avoid internal execution chatter, and only use tools when they materially improve accuracy or complete the task.',
        tools: ['web_search', 'web_fetch', 'query_memory'],
        defaultLoopMode: 'auto',
        defaultContextMode: 'v2',
        defaultUsePlanner: false,
        bootstrapFiles: ['IDENTITY.md', 'SOUL.md'],
    },
};


export const Dashboard: React.FC<DashboardProps> = ({ onSelectAgent, embedModels }) => {
    const [agents, setAgents] = useState<AgentProfile[]>([]);
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [agentToEdit, setAgentToEdit] = useState<AgentProfile | null>(null);
    const [agentToDelete, setAgentToDelete] = useState<AgentProfile | null>(null);
    const [loading, setLoading] = useState(true);
    const [availableModels, setAvailableModels] = useState<Record<string, string[]>>({});
    const [savingModel, setSavingModel] = useState<string | null>(null);
    const [agentSessions, setAgentSessions] = useState<Record<string, any[]>>({});

    useEffect(() => {
        fetchDashboardData();
        fetch('/api/models').then(r => r.json()).then(d => {
            if (d.models) setAvailableModels(d.models);
        }).catch(() => { });
    }, []);

    const fetchDashboardData = async () => {
        try {
            const [agentsRes, sessionsRes] = await Promise.all([
                fetch(withOwnerQuery('/api/agents')),
                fetch(withOwnerQuery('/api/sessions')).catch(() => ({ json: () => ({ sessions: [] }) }))
            ]);
            
            const agentsData = await agentsRes.json();
            const sessionsData = await sessionsRes.json();
            
            setAgents(agentsData.agents || []);
            
            const sessionsByAgent: Record<string, any[]> = {};
            if (sessionsData.sessions) {
                sessionsData.sessions.forEach((s: any) => {
                    const id = s.agent_id || 'default';
                    if (!sessionsByAgent[id]) sessionsByAgent[id] = [];
                    sessionsByAgent[id].push(s);
                });
            }
            
            for (const id in sessionsByAgent) {
                sessionsByAgent[id].sort((a, b) => new Date(b.updated_at || b.created_at).getTime() - new Date(a.updated_at || a.created_at).getTime());
            }
            
            setAgentSessions(sessionsByAgent);

        } catch (e) { console.error('Failed to fetch dashboard data:', e); }
        finally { setLoading(false); }
    };

    const handleModelChange = async (agentId: string, newModel: string) => {
        setAgents(prev => prev.map(a => a.id === agentId ? { ...a, model: newModel } : a));
        setSavingModel(agentId);
        try {
            await fetch(withOwnerQuery(`/api/agents/${agentId}`), {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(withOwnerPayload({ model: newModel })),
            });
        } catch (e) {
            console.error('Failed to update agent model:', e);
            fetchDashboardData();
        } finally {
            setSavingModel(null);
        }
    };

    const confirmDelete = async () => {
        if (!agentToDelete) return;
        try {
            const res = await fetch(withOwnerQuery(`/api/agents/${agentToDelete.id}`), { method: 'DELETE' });
            if (res.ok) fetchDashboardData();
            else {
                const data = await res.json();
                alert(data.detail || 'Failed to delete agent');
            }
        } catch (e) { console.error('Delete failed:', e); }
        finally { setAgentToDelete(null); }
    };

    if (loading) {
        return (
            <div className="dashboard-container">
                <div style={{ color: 'var(--text-dim)', fontSize: '12px', letterSpacing: '.2em', fontFamily: 'var(--mono)' }}>
                    INITIALIZING PLATFORM...
                </div>
            </div>
        );
    }

    return (
        <div className="dashboard-container">
            <header className="dashboard-header">
                <div className="branding-group">
                    <h1>Agent Workspace</h1>
                    <p>SHOVS // PLATFORM</p>
                </div>
                <button
                    className="btn-create-agent"
                    onClick={() => setShowCreateModal(true)}
                >
                    <span>+</span> INITIALIZE AGENT
                </button>
            </header>

            <div className="agent-grid">
                {agents.length === 0 ? (
                    <div className="dashboard-empty">
                        <div className="empty-visual">◈</div>
                        <h2>No Agents Optimized</h2>
                        <p>Begin by creating your first agentic workforce.</p>
                    </div>
                ) : (
                    agents.map(agent => (
                        <div key={agent.id} className="agent-card" onClick={() => onSelectAgent(agent.id)}>
                            <div className="agent-card-head">
                                <div className="agent-avatar">{agent.name.charAt(0).toUpperCase()}</div>
                                <div className="agent-stats">
                                    <div className="stat-pill">{agent.tools.length} TOOLS</div>
                                    {agentSessions[agent.id] && (
                                        <div className="stat-pill">{agentSessions[agent.id].length} CHATS</div>
                                    )}
                                </div>
                            </div>
                            
                            <div className="agent-info">
                                <h3>{agent.name}</h3>
                                <p>{agent.description || 'General purpose autonomous agent.'}</p>
                            </div>

                            <div className="agent-model-badge">
                                {agent.model}
                            </div>

                            <div className="agent-model-selector" onClick={e => e.stopPropagation()}>
                                <PremiumSelect
                                    label={savingModel === agent.id ? 'SYNCHRONIZING...' : 'MODEL'}
                                    value={agent.model}
                                    options={availableModels}
                                    onChange={(newModel) => handleModelChange(agent.id, newModel)}
                                />
                            </div>

                            <div className="agent-footer">
                                <button className="launch-btn" onClick={(e) => { e.stopPropagation(); onSelectAgent(agent.id); }}>
                                    LAUNCH WORKSPACE
                                </button>
                                <button className="btn-secondary" onClick={(e) => { e.stopPropagation(); setAgentToEdit(agent); }}>
                                    EDIT
                                </button>
                                {agent.id !== 'default' && (
                                    <button className="danger-btn" title="Delete Agent" onClick={(e) => { e.stopPropagation(); setAgentToDelete(agent); }}>
                                        ✕
                                    </button>
                                )}
                            </div>
                        </div>
                    ))
                )}
            </div>

            {showCreateModal && (
                <CreateAgentModal
                    embedModels={embedModels}
                    onClose={() => setShowCreateModal(false)}
                    onCreated={() => { setShowCreateModal(false); fetchDashboardData(); }}
                />
            )}
            {agentToEdit && (
                <CreateAgentModal
                    initialAgent={agentToEdit}
                    embedModels={embedModels}
                    onClose={() => setAgentToEdit(null)}
                    onCreated={() => { setAgentToEdit(null); fetchDashboardData(); }}
                />
            )}
            {agentToDelete && (
                <DeleteConfirmationModal
                    agentName={agentToDelete.name}
                    onClose={() => setAgentToDelete(null)}
                    onConfirm={confirmDelete}
                />
            )}
        </div>
    );
};


const CreateAgentModal: React.FC<{ onClose: () => void; onCreated: () => void; embedModels: Record<string, string[]>; initialAgent?: AgentProfile | null }> = ({ onClose, onCreated, embedModels, initialAgent }) => {
    const isEdit = Boolean(initialAgent);
    const [selectedPreset, setSelectedPreset] = useState<AgentPresetId | ''>('');
    const [name, setName] = useState(initialAgent?.name || '');
    const [description, setDescription] = useState(initialAgent?.description || '');
    const [model, setModel] = useState(initialAgent?.model || 'llama3.2');
    const [embedModel, setEmbedModel] = useState(initialAgent?.embed_model || 'nomic-embed-text');
    const [systemPrompt, setSystemPrompt] = useState(initialAgent?.system_prompt || '');
    const [workspacePath, setWorkspacePath] = useState(initialAgent?.workspace_path || '');
    const [bootstrapFilesText, setBootstrapFilesText] = useState((initialAgent?.bootstrap_files || ['AGENTS.md', 'IDENTITY.md', 'SOUL.md', 'TOOLS.md']).join(', '));
    const [bootstrapMaxChars, setBootstrapMaxChars] = useState(String(initialAgent?.bootstrap_max_chars || 8000));
    const [defaultUsePlanner, setDefaultUsePlanner] = useState(initialAgent?.default_use_planner ?? true);
    const [defaultLoopMode, setDefaultLoopMode] = useState<'auto' | 'single' | 'managed'>(initialAgent?.default_loop_mode || 'auto');
    const [defaultContextMode, setDefaultContextMode] = useState<'v1' | 'v2' | 'v3'>(initialAgent?.default_context_mode || 'v2');
    const [selectedTools, setSelectedTools] = useState<string[]>(initialAgent?.tools || ['web_search', 'web_fetch', 'query_memory', 'store_memory']);
    const [availableTools, setAvailableTools] = useState<any[]>([]);
    const [availableModels, setAvailableModels] = useState<Record<string, string[]>>({ 'ollama': ['llama3.2'] });
    const [creating, setCreating] = useState(false);

    useEffect(() => {
        fetch('/api/tools').then(r => r.json()).then(d => setAvailableTools(d.tools || []));
        fetch('/api/models').then(r => r.json()).then(d => {
            if (d.models) setAvailableModels(d.models);
        });
    }, []);

    const parsedBootstrapFiles = bootstrapFilesText
        .split(/[,\n]/)
        .map(v => v.trim())
        .filter(Boolean);

    const effectiveBootstrapBudget = Number(bootstrapMaxChars) || 8000;
    const estimatedPerDocBudget = parsedBootstrapFiles.length > 0
        ? Math.max(600, Math.floor(effectiveBootstrapBudget / parsedBootstrapFiles.length))
        : effectiveBootstrapBudget;

    const applyPreset = (presetId: AgentPresetId) => {
        const preset = AGENT_PRESETS[presetId];
        setSelectedPreset(presetId);
        if (!name.trim()) setName(presetId);
        if (!description.trim() || !isEdit) setDescription(preset.description);
        if (!systemPrompt.trim() || !isEdit) setSystemPrompt(preset.systemPrompt);
        setSelectedTools(preset.tools);
        setDefaultLoopMode(preset.defaultLoopMode);
        setDefaultContextMode(preset.defaultContextMode);
        setDefaultUsePlanner(preset.defaultUsePlanner);
        setBootstrapFilesText(preset.bootstrapFiles.join(', '));
    };

    const toggleTool = (name: string) => setSelectedTools(prev =>
        prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name]
    );

    const handleCreate = async () => {
        if (!name.trim()) return;
        setCreating(true);
        try {
            const payload = withOwnerPayload({
                name: name.trim(),
                description,
                model,
                embed_model: embedModel,
                system_prompt: systemPrompt,
                tools: selectedTools,
                workspace_path: workspacePath.trim() || null,
                bootstrap_files: bootstrapFilesText
                    .split(/[,\n]/)
                    .map(v => v.trim())
                    .filter(Boolean),
                bootstrap_max_chars: Number(bootstrapMaxChars) || 8000,
                default_use_planner: defaultUsePlanner,
                default_loop_mode: defaultLoopMode,
                default_context_mode: defaultContextMode,
            });
            await fetch(isEdit ? withOwnerQuery(`/api/agents/${initialAgent!.id}`) : '/api/agents', {
                method: isEdit ? 'PATCH' : 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            onCreated();
        } catch (e) { console.error('Create failed:', e); }
        finally { setCreating(false); }
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <h2>{isEdit ? 'Edit Agent' : 'Initialize Agent'}</h2>

                <div className="form-section">
                    <div className="input-group">
                        <label>Starting Templates</label>
                        <div className="agent-preset-row">
                            {(Object.keys(AGENT_PRESETS) as AgentPresetId[]).map((presetId) => (
                                <button
                                    key={presetId}
                                    type="button"
                                    className={`agent-preset-btn ${selectedPreset === presetId ? 'active' : ''}`}
                                    onClick={() => applyPreset(presetId)}
                                    title={AGENT_PRESETS[presetId].description}
                                >
                                    {AGENT_PRESETS[presetId].label}
                                </button>
                            ))}
                        </div>
                        <div className="agent-builder-note" style={{ marginTop: '8px' }}>
                            Templates are only starting points. The agent can be steered and molded later through chat, prompt edits, and capability changes.
                        </div>
                    </div>

                    <div className="input-group">
                        <label>Identifier</label>
                        <input
                            value={name}
                            onChange={e => setName(e.target.value)}
                            placeholder="e.g. research-assistant"
                            autoFocus
                        />
                    </div>

                    <div className="input-group">
                        <label>Mission Description</label>
                        <textarea
                            value={description}
                            onChange={e => setDescription(e.target.value)}
                            placeholder="What tasks should this agent prioritize?"
                            rows={3}
                        />
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                        <div className="input-group">
                            <PremiumSelect
                                label="Execution Model"
                                value={model}
                                options={availableModels}
                                onChange={setModel}
                            />
                        </div>

                        <div className="input-group">
                            <PremiumSelect
                                label="Embedding Model"
                                value={embedModel}
                                options={embedModels}
                                onChange={setEmbedModel}
                            />
                        </div>
                    </div>

                    <div className="input-group">
                        <label>System Prompt</label>
                        <textarea
                            value={systemPrompt}
                            onChange={e => setSystemPrompt(e.target.value)}
                            placeholder="Optional agent-specific operating prompt"
                            rows={5}
                        />
                    </div>

                    <div className="input-group">
                        <label>Core Capabilities</label>
                        <div className="tool-selection">
                            <div className="tool-chips">
                                {availableTools.map(t => (
                                    <span
                                        key={t.name}
                                        className={`chip ${selectedTools.includes(t.name) ? 'active' : ''}`}
                                        onClick={() => toggleTool(t.name)}
                                        title={t.description}
                                    >
                                        {t.name}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                        <div className="input-group">
                            <label>Default Loop</label>
                            <PremiumSelect
                                label="Execution Loop"
                                value={defaultLoopMode}
                                options={{ runtime: ['auto', 'single', 'managed'] }}
                                onChange={(value) => setDefaultLoopMode(value as 'auto' | 'single' | 'managed')}
                            />
                        </div>
                        <div className="input-group">
                            <label>Default Context Mode</label>
                            <PremiumSelect
                                label="Context Engine"
                                value={defaultContextMode}
                                options={{ context: ['v1', 'v2', 'v3'] }}
                                onChange={(value) => setDefaultContextMode(value as 'v1' | 'v2' | 'v3')}
                            />
                        </div>
                    </div>

                    <div className="input-group">
                        <label style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <input
                                type="checkbox"
                                checked={defaultUsePlanner}
                                onChange={e => setDefaultUsePlanner(e.target.checked)}
                            />
                            Default Planner
                        </label>
                    </div>

                    <div className="input-group">
                        <label>Workspace Path</label>
                        <input
                            value={workspacePath}
                            onChange={e => setWorkspacePath(e.target.value)}
                            placeholder="/absolute/path/to/agent-workspace"
                        />
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 180px', gap: '20px' }}>
                        <div className="input-group">
                            <label>Bootstrap Files</label>
                            <textarea
                                value={bootstrapFilesText}
                                onChange={e => setBootstrapFilesText(e.target.value)}
                                placeholder="AGENTS.md, IDENTITY.md, SOUL.md, TOOLS.md"
                                rows={3}
                            />
                        </div>
                        <div className="input-group">
                            <label>Bootstrap Budget</label>
                            <input
                                type="number"
                                min={1000}
                                max={20000}
                                step={500}
                                value={bootstrapMaxChars}
                                onChange={e => setBootstrapMaxChars(e.target.value)}
                            />
                        </div>
                    </div>

                    <div className="agent-builder-summary">
                        <div className="agent-builder-card">
                            <div className="agent-builder-card-label">Bootstrap Preview</div>
                            <div className="agent-builder-chip-row">
                                {parsedBootstrapFiles.length > 0 ? (
                                    parsedBootstrapFiles.map((file) => (
                                        <span key={file} className="agent-builder-chip">{file}</span>
                                    ))
                                ) : (
                                    <span className="agent-builder-empty">No bootstrap docs configured</span>
                                )}
                            </div>
                            <div className="agent-builder-note">
                                {workspacePath.trim()
                                    ? `Workspace root: ${workspacePath.trim()}`
                                    : 'No workspace path set. Bootstrap docs will only load if the runtime can resolve them from the configured workspace.'}
                            </div>
                        </div>

                        <div className="agent-builder-card">
                            <div className="agent-builder-card-label">Prompt Contribution Summary</div>
                            <div className="agent-builder-metrics">
                                <div><span>System Prompt</span><strong>{systemPrompt.trim() ? 'custom' : 'platform default'}</strong></div>
                                <div><span>Bootstrap Docs</span><strong>{parsedBootstrapFiles.length}</strong></div>
                                <div><span>Bootstrap Budget</span><strong>{effectiveBootstrapBudget} chars</strong></div>
                                <div><span>Per Doc Budget</span><strong>{estimatedPerDocBudget} chars</strong></div>
                                <div><span>Loop Mode</span><strong>{defaultLoopMode}</strong></div>
                                <div><span>Context Mode</span><strong>{defaultContextMode}</strong></div>
                                <div><span>Planner</span><strong>{defaultUsePlanner ? 'on' : 'off'}</strong></div>
                                <div><span>Selected Tools</span><strong>{selectedTools.length}</strong></div>
                            </div>
                            <div className="agent-builder-note">
                                Runtime shape: platform core prompt + agent prompt + selected bootstrap docs + tool registry + loop/context defaults.
                            </div>
                        </div>
                    </div>
                </div>

                <div className="modal-actions">
                    <button className="btn-secondary" onClick={onClose}>CANCEL</button>
                    <button
                        className="btn-primary"
                        onClick={handleCreate}
                        disabled={!name.trim() || creating}
                    >
                        {creating ? (isEdit ? 'SAVING...' : 'INITIALIZING...') : (isEdit ? 'SAVE AGENT' : 'INITIALIZE AGENT')}
                    </button>
                </div>
            </div>
        </div>
    );
};

const DeleteConfirmationModal: React.FC<{ agentName: string; onClose: () => void; onConfirm: () => void }> = ({ agentName, onClose, onConfirm }) => {
    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <h2 style={{ color: 'var(--error)' }}>Confirm Termination</h2>
                <p style={{ margin: '15px 0', color: 'var(--text-mid)', fontSize: '14px', lineHeight: '1.6' }}>
                    Are you sure you want to terminate <strong style={{ color: '#fff' }}>{agentName}</strong>?
                    This action is irreversible and will purge all neural configurations for this unit.
                </p>
                <div className="modal-actions">
                    <button className="btn-secondary" onClick={onClose}>ABORT</button>
                    <button className="danger-btn" style={{ width: 'auto', flex: 1, textTransform: 'uppercase', fontSize: '12px' }} onClick={onConfirm}>TERMINATE UNIT</button>
                </div>
            </div>
        </div>
    );
};
