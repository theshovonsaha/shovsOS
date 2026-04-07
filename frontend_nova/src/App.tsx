import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Dashboard } from './Dashboard';
import { LogPanel } from './LogPanel';
import { useAgent } from './useAgent';
import { GuardrailConfirmationModal } from './components/GuardrailConfirmationModal';
import { OptionsPanel } from './components/OptionsPanel';
import { PremiumSelect } from './components/PremiumSelect';
import { RichContentViewer } from './components/RichContentViewer';
import { ShovsView } from './components/ShovsView';
import { TraceMonitor } from './components/TraceMonitor';
import { VoiceControl } from './components/VoiceControl';

const MAX_TOOL_SUMMARY_LENGTH = 84;
const TRUNCATED_TOOL_SUMMARY_LENGTH = MAX_TOOL_SUMMARY_LENGTH - 3;

const TOOL_LABELS: Record<string, string> = {
  web_search: 'Searching the web',
  web_fetch: 'Reading page',
  rag_search: 'Searching session memory',
  query_memory: 'Checking long-term memory',
  store_memory: 'Saving memory',
  file_create: 'Creating file',
  file_view: 'Opening file',
  file_str_replace: 'Editing file',
  bash: 'Running command',
  generate_app: 'Building app',
};

const truncateToolSummary = (content: string) => {
  if (content.length <= MAX_TOOL_SUMMARY_LENGTH) {
    return content;
  }

  const truncated = content.slice(0, TRUNCATED_TOOL_SUMMARY_LENGTH);
  const boundary = truncated.lastIndexOf(' ');
  return `${(boundary > 24 ? truncated.slice(0, boundary) : truncated).trimEnd()}...`;
};

const parseToolPayload = (content?: string) => {
  if (!content) return null;

  try {
    const trimmed = content.trim();
    const start = trimmed.indexOf('{');
    const end = trimmed.lastIndexOf('}');
    if (start !== -1 && end !== -1 && end > start) {
      return JSON.parse(trimmed.substring(start, end + 1));
    }
  } catch {
    // Keep fallback rendering for plain text payloads.
  }

  return null;
};

const summarizeToolContent = (
  type: 'call' | 'result' | 'error',
  content?: string,
) => {
  const payload = parseToolPayload(content);
  let label =
    type === 'call' ? 'Working' : type === 'result' ? 'Ready' : 'Failed';
  let summary = '';
  let autoExpand = type === 'error';

  if (type === 'call') {
    summary = content || 'working...';
  } else if (
    payload?.type === 'web_search_results' &&
    Array.isArray(payload.results)
  ) {
    label = 'Sources Ready';
    summary = `${payload.results.length} source${payload.results.length === 1 ? '' : 's'} curated`;
  } else if (payload?.type === 'web_fetch_result') {
    label = payload.error ? 'Read Failed' : 'Page Ready';
    summary =
      payload.error ||
      payload.url ||
      `${payload.total_length || 0} chars loaded`;
  } else if (payload && (payload.type === 'app_view' || payload.path)) {
    label = 'Preview Ready';
    summary = payload.title || payload.path || 'Interactive sandbox preview';
  } else if (content) {
    summary = truncateToolSummary(content);
  }

  return { label, summary, autoExpand };
};

const ToolEvent = ({
  type,
  tool,
  content,
}: {
  type: 'call' | 'result' | 'error';
  tool: string;
  content?: string;
}) => {
  const { label, summary, autoExpand } = useMemo(
    () => summarizeToolContent(type, content),
    [type, content],
  );
  const [expanded, setExpanded] = useState(autoExpand);
  const canExpand = Boolean(content);

  useEffect(() => {
    if (autoExpand) {
      setExpanded(true);
    }
  }, [autoExpand]);

  const icon = type === 'call' ? '⚙' : type === 'result' ? '✓' : '✕';
  const statusText =
    type === 'call' ? 'running' : type === 'result' ? 'ready' : 'error';
  const friendlyName = TOOL_LABELS[tool] || tool.replace(/_/g, ' ');

  return (
    <div className={`nova-tool-event ${type} ${expanded ? 'expanded' : ''}`}>
      <div
        className='nova-tool-event-head'
        onClick={() => canExpand && setExpanded((prev) => !prev)}
      >
        <span className='nova-tool-event-icon'>{icon}</span>
        <span className='nova-tool-event-title'>
          <span className='nova-tool-event-label'>{label}</span>
          <span className='nova-tool-event-name'>{friendlyName}</span>
          {summary ? (
            <span className='nova-tool-event-summary'>{summary}</span>
          ) : null}
        </span>
        <span className={`nova-tool-event-state ${type}`}>{statusText}</span>
      </div>
      {expanded && canExpand ? (
        <div className='nova-tool-event-body'>
          <RichContentViewer content={content || ''} />
        </div>
      ) : null}
    </div>
  );
};

const ThoughtBlock = ({ content }: { content: string }) => {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className={`nova-thought ${expanded ? 'expanded' : ''}`}>
      <button
        className='nova-thought-head'
        onClick={() => setExpanded((prev) => !prev)}
      >
        <span>Reasoning</span>
        <span>{expanded ? '▴' : '▾'}</span>
      </button>
      {expanded ? (
        <div className='nova-thought-body'>
          <RichContentViewer content={content} />
        </div>
      ) : null}
    </div>
  );
};

const PlanBlock = ({ content }: { content: string }) => {
  const [expanded, setExpanded] = useState(true);
  return (
    <div className={`nova-plan ${expanded ? 'expanded' : ''}`}>
      <button
        className='nova-plan-head'
        onClick={() => setExpanded((prev) => !prev)}
      >
        <span>Strategy</span>
        <span>{expanded ? '▴' : '▾'}</span>
      </button>
      {expanded ? (
        <div className='nova-plan-body'>
          <RichContentViewer content={content} />
        </div>
      ) : null}
    </div>
  );
};

const TensionBlock = ({ content }: { content: string }) => {
  const [expanded, setExpanded] = useState(true);
  return (
    <div className={`nova-plan expanded tension`}>
      <button
        className='nova-plan-head'
        onClick={() => setExpanded((prev) => !prev)}
      >
        <span>Contradiction / Tension</span>
        <span>{expanded ? '▴' : '▾'}</span>
      </button>
      {expanded ? (
        <div className='nova-plan-body'>
          <RichContentViewer content={content} />
        </div>
      ) : null}
    </div>
  );
};

function App() {
  const agent = useAgent();
  const [inputText, setInputText] = useState('');
  const [logOpen, setLogOpen] = useState(false);
  const [isMobile, setIsMobile] = useState<boolean>(() =>
    typeof window !== 'undefined' ? window.innerWidth <= 760 : false,
  );
  const [mobileUtilitiesOpen, setMobileUtilitiesOpen] = useState(false);
  const [mobilePanel, setMobilePanel] = useState<
    'none' | 'sessions' | 'options'
  >('none');
  const [workspaceView, setWorkspaceView] = useState<'chat' | 'monitor'>(() => {
    const saved = localStorage.getItem('nova_workspace_view');
    return saved === 'monitor' ? 'monitor' : 'chat';
  });
  const [shovsMode, setShovsMode] = useState(false);
  const [toolMenuOpen, setToolMenuOpen] = useState(false);
  const [sidebarTab, setSidebarTab] = useState<'sessions' | 'options'>(
    'sessions',
  );

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const toolMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    localStorage.setItem('nova_workspace_view', workspaceView);
  }, [workspaceView]);

  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth <= 760);
    onResize();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  useEffect(() => {
    if (!isMobile) {
      setMobileUtilitiesOpen(false);
      setMobilePanel('none');
    }
  }, [isMobile]);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, 220)}px`;
  }, [inputText]);

  useEffect(() => {
    const onDocClick = (evt: MouseEvent) => {
      if (!toolMenuRef.current) return;
      if (!toolMenuRef.current.contains(evt.target as Node)) {
        setToolMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, []);

  const handleSend = () => {
    if (!inputText.trim() && agent.pendingFiles.length === 0) return;
    agent.sendMessage(inputText);
    setInputText('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const connectionLabel = agent.isStreaming
    ? 'answering'
    : agent.health.status === 'ok'
      ? 'connected'
      : 'connecting';

  if (!agent.activeAgentId) {
    return (
      <Dashboard
        onSelectAgent={(id) => agent.setActiveAgentId(id)}
        embedModels={agent.embedModels}
      />
    );
  }

  const sessionSidebar = (
    <>
      <div className='nova-sidebar-head'>
        <span>Conversation Threads</span>
        <button
          onClick={() => {
            agent.newSession();
            if (isMobile) setMobilePanel('none');
          }}
        >
          New
        </button>
      </div>
      <div className='nova-session-list'>
        {agent.sessions.length === 0 ? (
          <div className='nova-empty-card'>No sessions yet.</div>
        ) : (
          agent.sessions.map((session) => (
            <button
              key={session.id}
              className={`nova-session-item ${session.id === agent.currentSessionId ? 'active' : ''}`}
              onClick={() => {
                agent.loadSession(session.id);
                if (isMobile) setMobilePanel('none');
              }}
            >
              <div className='nova-session-item-main'>
                <span className='nova-session-title'>
                  {session.title || 'New Chat'}
                </span>
                <span className='nova-session-meta'>
                  {session.message_count} msg ·{' '}
                  {(session.model || '').split(':')[0]}
                </span>
              </div>
              <span
                className='nova-session-delete'
                onClick={(e) => {
                  e.stopPropagation();
                  agent.deleteSession(session.id);
                }}
              >
                ✕
              </span>
            </button>
          ))
        )}
      </div>

      {!isMobile ? (
        <div className='nova-stats-card'>
          <div className='nova-stats-row'>
            <span>Context</span>
            <span>
              {agent.contextLines > 0 ? `${agent.contextLines} items` : 'cold'}
            </span>
          </div>
          <div className='nova-stats-row'>
            <span>Tools</span>
            <span>{agent.tools.length}</span>
          </div>
          <div className='nova-tools-cloud'>
            {agent.tools.slice(0, 10).map((tool) => (
              <span key={tool.name}>{tool.name}</span>
            ))}
          </div>
        </div>
      ) : null}
    </>
  );

  const controlsSidebar = (
    <OptionsPanel
      sessionId={agent.currentSessionId}
      contextLines={agent.contextLines}
      currentSearchEngine={agent.currentSearchEngine}
      setCurrentSearchEngine={agent.setCurrentSearchEngine}
      models={agent.models}
      runtimePath={agent.runtimePath}
      setRuntimePath={agent.setRuntimePath}
      usePlanner={agent.usePlanner}
      setUsePlanner={agent.setUsePlanner}
      loopMode={agent.loopMode}
      setLoopMode={agent.setLoopMode}
      maxToolCalls={agent.maxToolCalls}
      setMaxToolCalls={agent.setMaxToolCalls}
      maxTurns={agent.maxTurns}
      setMaxTurns={agent.setMaxTurns}
      plannerModel={agent.plannerModel}
      setPlannerModel={agent.setPlannerModel}
      contextModel={agent.contextModel}
      setContextModel={agent.setContextModel}
      embedModel={agent.embedModel}
      setEmbedModel={agent.setEmbedModel}
      embedModels={agent.embedModels}
      contextMode={agent.contextMode}
      setSessionContextMode={agent.setSessionContextMode}
      clearSessionContext={agent.clearSessionContext}
      showPlannerLog={agent.showPlannerLog}
      setShowPlannerLog={agent.setShowPlannerLog}
      showActorThought={agent.showActorThought}
      setShowActorThought={agent.setShowActorThought}
      showObserverActivity={agent.showObserverActivity}
      setShowObserverActivity={agent.setShowObserverActivity}
    />
  );

  return (
    <div className={`nova-shell ${logOpen ? 'nova-log-open' : ''}`}>
      <header className='nova-topbar'>
        <div className='nova-topbar-left'>
          <button
            className='nova-back-btn'
            onClick={() => agent.setActiveAgentId(null)}
          >
            Agents
          </button>
          <div className='nova-brand'>
            <span className='nova-brand-main'>NOVA</span>
            <span className='nova-brand-sub'>agent workspace</span>
          </div>
        </div>

        <div className='nova-workspace-switch'>
          <button
            className={workspaceView === 'chat' ? 'active' : ''}
            onClick={() => setWorkspaceView('chat')}
          >
            Chat
          </button>
          <button
            className={workspaceView === 'monitor' ? 'active' : ''}
            onClick={() => setWorkspaceView('monitor')}
          >
            Monitor
          </button>
        </div>

        <div className='nova-topbar-right'>
          <span
            className={`nova-connection ${agent.health.status === 'ok' ? 'ok' : 'cold'}`}
          >
            {connectionLabel}
          </span>
          {isMobile ? (
            <button
              className={`nova-ghost-btn ${mobileUtilitiesOpen ? 'active' : ''}`}
              onClick={() => setMobileUtilitiesOpen((prev) => !prev)}
            >
              {mobileUtilitiesOpen ? 'Hide' : 'Quick'}
            </button>
          ) : (
            <>
              <PremiumSelect
                value={agent.currentModel}
                options={agent.models}
                onChange={(m) => agent.setCurrentModel(m)}
                placeholder='Select model'
              />
              <button
                className={`nova-ghost-btn ${agent.showActorThought ? 'active' : ''}`}
                onClick={() =>
                  agent.setShowActorThought(!agent.showActorThought)
                }
              >
                {agent.showActorThought ? 'Reasoning On' : 'Reasoning Off'}
              </button>
              <button
                className='nova-ghost-btn'
                onClick={() => setLogOpen((prev) => !prev)}
              >
                {logOpen ? 'Hide Logs' : 'Logs'}
              </button>
              <button
                className='nova-ghost-btn'
                onClick={() => setShovsMode((prev) => !prev)}
              >
                {shovsMode ? 'Close Voice' : 'Voice HUD'}
              </button>
            </>
          )}
        </div>
      </header>

      {isMobile ? (
        <>
          <div
            className={`nova-mobile-tray ${mobileUtilitiesOpen ? 'open' : ''}`}
          >
            <PremiumSelect
              value={agent.currentModel}
              options={agent.models}
              onChange={(m) => agent.setCurrentModel(m)}
              placeholder='Select model'
            />
            <div className='nova-mobile-tray-actions'>
              <button
                className={`nova-ghost-btn ${agent.showActorThought ? 'active' : ''}`}
                onClick={() =>
                  agent.setShowActorThought(!agent.showActorThought)
                }
              >
                {agent.showActorThought ? 'Reasoning On' : 'Reasoning Off'}
              </button>
              <button
                className='nova-ghost-btn'
                onClick={() => setLogOpen((prev) => !prev)}
              >
                {logOpen ? 'Hide Logs' : 'Logs'}
              </button>
              <button
                className='nova-ghost-btn'
                onClick={() => setShovsMode((prev) => !prev)}
              >
                {shovsMode ? 'Close Voice' : 'Voice HUD'}
              </button>
            </div>
          </div>
          <div className='nova-mobile-rail'>
            <button
              className={mobilePanel === 'sessions' ? 'active' : ''}
              onClick={() => {
                setSidebarTab('sessions');
                setMobilePanel((prev) =>
                  prev === 'sessions' ? 'none' : 'sessions',
                );
              }}
            >
              Threads
            </button>
            <button
              className={mobilePanel === 'options' ? 'active' : ''}
              onClick={() => {
                setSidebarTab('options');
                setMobilePanel((prev) =>
                  prev === 'options' ? 'none' : 'options',
                );
              }}
            >
              Controls
            </button>
            <button onClick={agent.newSession}>New</button>
            <button onClick={() => setMobileUtilitiesOpen((prev) => !prev)}>
              Tools
            </button>
          </div>
        </>
      ) : null}

      <div className='nova-body'>
        {!isMobile ? (
          <aside className='nova-sidebar'>
            <div className='nova-sidebar-tabs'>
              <button
                className={sidebarTab === 'sessions' ? 'active' : ''}
                onClick={() => setSidebarTab('sessions')}
              >
                Sessions
              </button>
              <button
                className={sidebarTab === 'options' ? 'active' : ''}
                onClick={() => setSidebarTab('options')}
              >
                Controls
              </button>
            </div>

            {sidebarTab === 'sessions' ? sessionSidebar : controlsSidebar}
          </aside>
        ) : null}

        <main className='nova-main'>
          {workspaceView === 'monitor' ? (
            <TraceMonitor
              sessionId={agent.currentSessionId}
              isVisible={workspaceView === 'monitor'}
            />
          ) : (
            <>
              <section
                className='nova-conversation'
                ref={agent.conversationRef}
              >
                {agent.messages.length === 0 ? (
                  <div className='nova-conversation-empty'>
                    <h2>Minimal surface, maximum visibility.</h2>
                    <p>
                      Start with a prompt. Deep logs, trace telemetry, and
                      safety approvals remain available when needed.
                    </p>
                  </div>
                ) : (
                  agent.messages.map((message, idx) => (
                    <article
                      key={message.id || idx}
                      className={`nova-message ${message.role}`}
                    >
                      <div className='nova-message-role'>
                        {message.role === 'user' ? 'You' : 'Agent'}
                      </div>
                      <div className='nova-message-body'>
                        {message.files?.filter((f) => f.dataURL).length ? (
                          <div className='nova-image-row'>
                            {message
                              .files!.filter((f) => f.dataURL)
                              .map((f) => (
                                <img
                                  key={f.id}
                                  className='nova-image-chip'
                                  src={f.dataURL!}
                                  title={f.file.name}
                                  alt='attachment'
                                />
                              ))}
                          </div>
                        ) : null}

                        {message.files?.filter((f) => !f.dataURL).length ? (
                          <div className='nova-file-row'>
                            {message
                              .files!.filter((f) => !f.dataURL)
                              .map((f) => (
                                <span key={f.id} className='nova-file-chip'>
                                  {f.file.name}
                                </span>
                              ))}
                          </div>
                        ) : null}

                        {message.role === 'user' && !message.blocks?.length ? (
                          <RichContentViewer content={message.content} />
                        ) : null}

                        {message.blocks?.map((block) => {
                          switch (block.type) {
                            case 'text':
                              return (
                                <RichContentViewer
                                  key={block.id}
                                  content={block.content}
                                />
                              );
                            case 'thought':
                              return agent.showActorThought ? (
                                <ThoughtBlock
                                  key={block.id}
                                  content={block.content}
                                />
                              ) : null;
                            case 'plan':
                              return agent.showPlannerLog ? (
                                <PlanBlock
                                  key={block.id}
                                  content={block.content}
                                />
                              ) : null;
                            case 'tension_hint':
                              return agent.showObserverActivity ? (
                                <TensionBlock
                                  key={block.id}
                                  content={block.content}
                                />
                              ) : null;
                            case 'tool_call':
                              return (
                                <ToolEvent
                                  key={block.id}
                                  type='call'
                                  tool={block.tool || 'unknown'}
                                  content={block.content || ''}
                                />
                              );
                            case 'tool_result':
                              return (
                                <ToolEvent
                                  key={block.id}
                                  type='result'
                                  tool={block.tool || 'unknown'}
                                  content={block.content || ''}
                                />
                              );
                            case 'tool_error':
                              return (
                                <ToolEvent
                                  key={block.id}
                                  type='error'
                                  tool={block.tool || 'unknown'}
                                  content={block.content || ''}
                                />
                              );
                            case 'attachment_badge':
                              return (
                                <div
                                  key={block.id}
                                  className='nova-inline-badge'
                                >
                                  {block.content}
                                </div>
                              );
                            case 'compressing':
                              return (
                                <div
                                  key={block.id}
                                  className='nova-inline-badge subtle'
                                >
                                  compressing context...
                                </div>
                              );
                            default:
                              return null;
                          }
                        })}

                        {agent.isStreaming &&
                        message.role === 'assistant' &&
                        idx === agent.messages.length - 1 ? (
                          <span className='nova-cursor' />
                        ) : null}
                      </div>
                    </article>
                  ))
                )}
                <div ref={agent.bottomRef} />
              </section>

              <section
                className='nova-composer'
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  if (e.dataTransfer.files) {
                    agent.addFiles(Array.from(e.dataTransfer.files));
                  }
                }}
              >
                <div className='nova-chip-row'>
                  {agent.forcedTools.map((toolName) => (
                    <span key={`forced-${toolName}`} className='nova-chip tool'>
                      <span>{toolName}</span>
                      <button
                        onClick={() =>
                          agent.setForcedTools((prev) =>
                            prev.filter((name) => name !== toolName),
                          )
                        }
                      >
                        ✕
                      </button>
                    </span>
                  ))}

                  {agent.pendingFiles.map((file) => (
                    <span key={file.id} className='nova-chip'>
                      <span>{file.file.name}</span>
                      <button onClick={() => agent.removeFile(file.id)}>
                        ✕
                      </button>
                    </span>
                  ))}
                </div>

                <div className='nova-composer-row'>
                  <VoiceControl
                    isRecording={agent.isListening}
                    status={agent.voiceStatus}
                    onToggle={() => {
                      if (agent.isListening) agent.stopRecording();
                      else if (agent.speaking) agent.stopSpeaking();
                      else agent.startRecording();
                    }}
                  />

                  <textarea
                    ref={textareaRef}
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    rows={1}
                    placeholder='Ask, build, inspect, orchestrate...'
                  />

                  <div className='nova-tool-menu-wrap' ref={toolMenuRef}>
                    <button
                      className={`nova-menu-btn ${toolMenuOpen ? 'active' : ''}`}
                      onClick={() => setToolMenuOpen((prev) => !prev)}
                    >
                      Tools
                    </button>

                    {toolMenuOpen ? (
                      <div className='nova-tool-menu'>
                        {agent.tools.map((tool) => {
                          const isSelected = agent.forcedTools.includes(
                            tool.name,
                          );
                          return (
                            <button
                              key={tool.name}
                              className={isSelected ? 'selected' : ''}
                              onClick={() => {
                                agent.setForcedTools((prev) =>
                                  prev.includes(tool.name)
                                    ? prev.filter((name) => name !== tool.name)
                                    : [...prev, tool.name],
                                );
                              }}
                            >
                              <span>{tool.name}</span>
                              <small>{tool.description.split('.')[0]}</small>
                            </button>
                          );
                        })}
                      </div>
                    ) : null}
                  </div>

                  <label className='nova-menu-btn file'>
                    Attach
                    <input
                      type='file'
                      multiple
                      onChange={(e) => {
                        if (e.target.files) {
                          agent.addFiles(Array.from(e.target.files));
                        }
                      }}
                      style={{ display: 'none' }}
                    />
                  </label>

                  <button
                    className={`nova-send-btn ${agent.isStreaming ? 'nova-stop-btn' : ''}`}
                    onClick={
                      agent.isStreaming ? agent.stopExecution : handleSend
                    }
                    disabled={
                      !agent.isStreaming &&
                      !inputText.trim() &&
                      agent.pendingFiles.length === 0
                    }
                  >
                    {agent.isStreaming ? 'Stop' : 'Send'}
                  </button>
                </div>

                <div className='nova-composer-foot'>
                  <span>Enter to send · Shift+Enter for new line</span>
                  <span>
                    Context{' '}
                    {agent.contextLines > 0
                      ? `${agent.contextLines} items`
                      : 'cold'}
                  </span>
                </div>
              </section>
            </>
          )}
        </main>
      </div>

      {isMobile && mobilePanel !== 'none' ? (
        <div
          className='nova-mobile-panel'
          onClick={() => setMobilePanel('none')}
        >
          <div
            className='nova-mobile-panel-sheet'
            onClick={(e) => e.stopPropagation()}
          >
            <div className='nova-mobile-panel-head'>
              <span>{mobilePanel === 'sessions' ? 'Threads' : 'Controls'}</span>
              <button onClick={() => setMobilePanel('none')}>Close</button>
            </div>
            <div className='nova-mobile-panel-body'>
              {mobilePanel === 'sessions' ? sessionSidebar : controlsSidebar}
            </div>
          </div>
        </div>
      ) : null}

      <LogPanel
        sessionId={agent.currentSessionId}
        isOpen={logOpen}
        onClose={() => setLogOpen(false)}
      />

      {shovsMode ? (
        <ShovsView
          onClose={() => setShovsMode(false)}
          isListening={agent.isListening}
          isThinking={agent.isStreaming && !agent.speaking}
          isSpeaking={agent.speaking}
          lastUserText={agent.lastUserText}
          currentAgentToken={agent.currentToken}
          lastAgentResponse={agent.lastAgentResponse}
          voiceSensitivity={agent.voiceSensitivity}
          setVoiceSensitivity={agent.setVoiceSensitivity}
          voiceModel={agent.voiceModel}
          setVoiceModel={agent.setVoiceModel}
          onToggleMic={() => {
            if (agent.isListening) {
              agent.stopRecording();
            } else if (agent.speaking) {
              agent.stopSpeaking();
            } else {
              agent.startRecording();
            }
          }}
        />
      ) : null}

      {agent.pendingConfirmation ? (
        <GuardrailConfirmationModal
          confirmation={agent.pendingConfirmation}
          onApprove={agent.approveConfirmation}
          onDeny={agent.denyConfirmation}
        />
      ) : null}
    </div>
  );
}

export default App;
