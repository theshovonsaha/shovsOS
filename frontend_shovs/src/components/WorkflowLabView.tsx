import React, { useEffect, useMemo, useState } from 'react';

interface WorkflowStep {
  id: string;
  label: string;
  kind: string;
  description: string;
  consumes: string[];
  produces: string[];
  gates: string[];
}

interface WorkflowDefinition {
  id: string;
  label: string;
  description: string;
  template: string;
  runtime: {
    policy: string;
    ledger_mode: string;
    context_mode: string;
    prompt_version: string;
    risk_policy: string;
  };
  tools: string[];
  memory_policy: Record<string, unknown>;
  input_schema: Record<string, string>;
  output_schema: {
    type: string;
    fields: string[];
  };
  steps: WorkflowStep[];
  tags: string[];
}

interface WorkflowCatalog {
  title: string;
  subtitle: string;
  workflows: WorkflowDefinition[];
  lifecycle: string[];
  run_modes?: string[];
  api: Record<string, string>;
  legal_usage: string[];
}

interface WorkflowEvent {
  id: string;
  phase: string;
  event_type: string;
  status: string;
  summary: string;
  created_at: string;
  data?: Record<string, unknown>;
}

interface WorkflowRunStatus {
  run_id: string;
  workflow_id: string;
  status: string;
  mode: string;
  created_at: string;
  updated_at: string;
  event_count: number;
  result_ready: boolean;
  error?: string;
  definition: WorkflowDefinition;
}

interface WorkflowRunResult {
  run_id: string;
  workflow_id: string;
  status: string;
  output_schema: {
    type: string;
    fields: string[];
  };
  summary: string;
  fields: Record<string, { status: string; value: string }>;
  api_contract: Record<string, string>;
  trace_summary: {
    event_count: number;
    policy: string;
    ledger_mode: string;
    tools: string[];
    live_run_engine?: boolean;
  };
}

type WorkflowTab = 'build' | 'run' | 'events' | 'result' | 'api';

interface WorkflowImageInput {
  id: string;
  name: string;
  size: number;
  dataUrl: string;
  base64: string;
}

const WORKFLOW_EXAMPLES: Record<string, Record<string, unknown>> = {
  research_brief_v1: {
    query: 'agent workflow traceability',
    source_count: 3,
    freshness: 'recent',
  },
  shopping_comparison_v1: {
    product: 'air purifier',
    location: 'Toronto',
    stores: ['Costco', 'Walmart', 'Best Buy'],
    budget: 'under $300',
  },
  local_recommendation_v1: {
    category: 'sushi',
    location: 'Toronto',
    constraints: 'quiet, high quality, dinner',
  },
  memory_fact_guard_v1: {
    claim: 'User prefers concise workflow dashboards.',
    source: 'explicit user statement',
  },
  coding_patch_eval_v1: {
    task: 'Add a focused API test for workflow status.',
    workspace_path: '/workspace/project',
    test_command: 'pytest tests/test_workflow_lab_api.py',
  },
  vision_inspection_v1: {
    question: 'What is visible in this image?',
    focus: 'objects, text, and obvious UI issues',
  },
};

const formatJson = (value: unknown) => JSON.stringify(value, null, 2);
const formatBytes = (bytes: number): string => {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB'];
  let index = 0;
  let value = bytes;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[index]}`;
};

const parseJsonObject = (text: string): Record<string, unknown> => {
  const parsed = JSON.parse(text);
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Input must be a JSON object.');
  }
  return parsed as Record<string, unknown>;
};

export const WorkflowLabView: React.FC = () => {
  const [catalog, setCatalog] = useState<WorkflowCatalog | null>(null);
  const [selectedWorkflowId, setSelectedWorkflowId] = useState('shopping_comparison_v1');
  const [selectedStepId, setSelectedStepId] = useState('intake');
  const [activeTab, setActiveTab] = useState<WorkflowTab>('build');
  const [inputText, setInputText] = useState(formatJson(WORKFLOW_EXAMPLES.shopping_comparison_v1));
  const [runMode, setRunMode] = useState('deterministic_contract');
  const [runStatus, setRunStatus] = useState<WorkflowRunStatus | null>(null);
  const [events, setEvents] = useState<WorkflowEvent[]>([]);
  const [result, setResult] = useState<WorkflowRunResult | null>(null);
  const [workflowImages, setWorkflowImages] = useState<WorkflowImageInput[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError('');
      try {
        const response = await fetch('/api/workflow-lab/catalog');
        if (!response.ok) throw new Error(`catalog failed: ${response.status}`);
        const payload = (await response.json()) as WorkflowCatalog;
        if (!cancelled) {
          setCatalog(payload);
          const first = payload.workflows.find((item) => item.id === selectedWorkflowId) || payload.workflows[0];
          if (first) {
            setSelectedWorkflowId(first.id);
            setSelectedStepId(first.steps[0]?.id || '');
            setInputText(formatJson(WORKFLOW_EXAMPLES[first.id] || {}));
          }
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load workflows.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedWorkflow = useMemo(
    () => catalog?.workflows.find((workflow) => workflow.id === selectedWorkflowId) || catalog?.workflows[0],
    [catalog, selectedWorkflowId],
  );
  const selectedStep =
    selectedWorkflow?.steps.find((step) => step.id === selectedStepId) ||
    selectedWorkflow?.steps[0];

  const selectWorkflow = (workflow: WorkflowDefinition) => {
    setSelectedWorkflowId(workflow.id);
    setSelectedStepId(workflow.steps[0]?.id || '');
    setInputText(formatJson(WORKFLOW_EXAMPLES[workflow.id] || {}));
    setRunStatus(null);
    setEvents([]);
    setResult(null);
    setWorkflowImages([]);
    setActiveTab('build');
    setError('');
  };
  const selectedWorkflowAcceptsImages = Boolean(
    selectedWorkflow?.tags.includes('vision') || selectedWorkflow?.input_schema.images,
  );

  const apiRequest = selectedWorkflow
    ? {
        method: 'POST',
        url: `/workflow-lab/workflows/${selectedWorkflow.id}/runs`,
        body: {
          input: (() => {
            try {
              const parsed = parseJsonObject(inputText);
              if (selectedWorkflowAcceptsImages && workflowImages.length) {
                return { ...parsed, image_count: workflowImages.length };
              }
              return parsed;
            } catch {
              return WORKFLOW_EXAMPLES[selectedWorkflow.id] || {};
            }
          })(),
          mode: runMode,
          images: selectedWorkflowAcceptsImages && workflowImages.length
            ? workflowImages.map((image) => image.base64)
            : undefined,
        },
      }
    : null;

  const addWorkflowImages = (fileList: FileList | File[]) => {
    const imageFiles = Array.from(fileList).filter((file) => file.type.startsWith('image/'));
    if (!imageFiles.length) return;
    imageFiles.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const dataUrl = String(event.target?.result || '');
        const base64 = dataUrl.includes(',') ? dataUrl.split(',').pop() || '' : dataUrl;
        if (!base64) return;
        setWorkflowImages((prev) => [
          ...prev,
          {
            id: `${file.name}-${file.size}-${Date.now()}-${Math.random().toString(36).slice(2)}`,
            name: file.name,
            size: file.size,
            dataUrl,
            base64,
          },
        ]);
      };
      reader.readAsDataURL(file);
    });
  };

  const loadRun = async (runId: string) => {
    const [statusResponse, eventsResponse] = await Promise.all([
      fetch(`/api/workflow-lab/runs/${runId}`),
      fetch(`/api/workflow-lab/runs/${runId}/events`),
    ]);
    if (!statusResponse.ok || !eventsResponse.ok) {
      throw new Error('run created but status fetch failed');
    }
    const status = (await statusResponse.json()) as WorkflowRunStatus;
    const eventPayload = (await eventsResponse.json()) as { events: WorkflowEvent[] };
    setRunStatus(status);
    setEvents(eventPayload.events || []);
    if (status.result_ready) {
      const resultResponse = await fetch(`/api/workflow-lab/runs/${runId}/result`);
      if (resultResponse.ok) {
        setResult((await resultResponse.json()) as WorkflowRunResult);
      }
    }
    return status;
  };

  const pollRun = async (runId: string) => {
    let latest: WorkflowRunStatus | null = null;
    for (let attempt = 0; attempt < 30; attempt += 1) {
      latest = await loadRun(runId);
      if (latest.status === 'completed' || latest.status === 'failed' || latest.result_ready) {
        return latest;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 800));
    }
    return latest;
  };

  const runWorkflow = async () => {
    if (!selectedWorkflow) return;
    setRunning(true);
    setError('');
    setCopied(false);
    try {
      const input = parseJsonObject(inputText);
      const payload: Record<string, unknown> = { input, mode: runMode };
      if (selectedWorkflowAcceptsImages && workflowImages.length) {
        payload.images = workflowImages.map((image) => image.base64);
      }
      const created = await fetch(`/api/workflow-lab/workflows/${selectedWorkflow.id}/runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!created.ok) throw new Error(`run failed: ${created.status}`);
      const run = (await created.json()) as { run_id: string };
      await pollRun(run.run_id);
      setActiveTab('events');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Workflow run failed.');
    } finally {
      setRunning(false);
    }
  };

  const copyApiRequest = async () => {
    if (!apiRequest) return;
    const text = `fetch("${apiRequest.url}", {\n  method: "${apiRequest.method}",\n  headers: { "Content-Type": "application/json" },\n  body: JSON.stringify(${formatJson(apiRequest.body)})\n})`;
    await navigator.clipboard?.writeText(text);
    setCopied(true);
  };

  if (loading) {
    return (
      <section className='workflow-lab-view'>
        <div className='workflow-empty'>Loading workflows...</div>
      </section>
    );
  }

  if (!catalog || !selectedWorkflow) {
    return (
      <section className='workflow-lab-view'>
        <div className='workflow-empty error'>{error || 'Workflow Lab is not available.'}</div>
      </section>
    );
  }

  return (
    <section className='workflow-lab-view'>
      <div className='workflow-lab-header'>
        <div>
          <div className='workspace-card-kicker'>Workflow Platform</div>
          <h2>{catalog.title}</h2>
          <p>{catalog.subtitle}</p>
        </div>
        <button className='shovs-send-btn' onClick={runWorkflow} disabled={running}>
          {running ? 'Running...' : 'Run Workflow'}
        </button>
      </div>

      {error ? <div className='workflow-error'>{error}</div> : null}

      <div className='workflow-lab-grid'>
        <aside className='workflow-list'>
          {catalog.workflows.map((workflow) => (
            <button
              key={workflow.id}
              className={workflow.id === selectedWorkflow.id ? 'active' : ''}
              onClick={() => selectWorkflow(workflow)}
            >
              <strong>{workflow.label}</strong>
              <span>{workflow.runtime.policy} · {workflow.runtime.ledger_mode}</span>
              <small>{workflow.tags.join(' · ')}</small>
            </button>
          ))}
        </aside>

        <main className='workflow-main-panel'>
          <div className='workflow-summary-row'>
            <div>
              <span>Template</span>
              <strong>{selectedWorkflow.template}</strong>
            </div>
            <div>
              <span>Policy</span>
              <strong>{selectedWorkflow.runtime.policy}</strong>
            </div>
            <div>
              <span>Context</span>
              <strong>{selectedWorkflow.runtime.context_mode}</strong>
            </div>
            <div>
              <span>Status</span>
              <strong>{runStatus?.status || 'not run'}</strong>
            </div>
            <div>
              <span>Mode</span>
              <strong>{runStatus?.mode || runMode}</strong>
            </div>
          </div>

          <div className='workflow-tabs' role='tablist' aria-label='Workflow Lab sections'>
            {(['build', 'run', 'events', 'result', 'api'] as WorkflowTab[]).map((tab) => (
              <button key={tab} className={activeTab === tab ? 'active' : ''} onClick={() => setActiveTab(tab)}>
                {tab}
              </button>
            ))}
          </div>

          {activeTab === 'build' ? (
            <div className='workflow-builder-grid'>
              <div className='workflow-step-timeline'>
                {selectedWorkflow.steps.map((step, index) => (
                  <button
                    key={step.id}
                    className={selectedStep?.id === step.id ? 'active' : ''}
                    onClick={() => setSelectedStepId(step.id)}
                  >
                    <span>{String(index + 1).padStart(2, '0')}</span>
                    <strong>{step.label}</strong>
                    <small>{step.kind}</small>
                  </button>
                ))}
              </div>
              {selectedStep ? (
                <div className='workflow-step-detail'>
                  <h3>{selectedStep.label}</h3>
                  <p>{selectedStep.description}</p>
                  <div className='workflow-detail-grid'>
                    <div>
                      <span>Consumes</span>
                      <strong>{selectedStep.consumes.join(', ') || 'none'}</strong>
                    </div>
                    <div>
                      <span>Produces</span>
                      <strong>{selectedStep.produces.join(', ') || 'none'}</strong>
                    </div>
                    <div>
                      <span>Gates</span>
                      <strong>{selectedStep.gates.join(', ') || 'none'}</strong>
                    </div>
                  </div>
                  <div className='workflow-tool-strip'>
                    {selectedWorkflow.tools.map((tool) => <span key={tool}>{tool}</span>)}
                  </div>
                </div>
              ) : null}
            </div>
          ) : null}

          {activeTab === 'run' ? (
            <div className='workflow-run-panel'>
              <div className='workflow-run-controls'>
                <label htmlFor='workflow-run-mode'>Run mode</label>
                <select
                  id='workflow-run-mode'
                  value={runMode}
                  onChange={(event) => setRunMode(event.target.value)}
                >
                  {(catalog.run_modes || ['deterministic_contract', 'live_run_engine']).map((mode) => (
                    <option key={mode} value={mode}>{mode}</option>
                  ))}
                </select>
                <button
                  className='shovs-ghost-btn'
                  onClick={() => runStatus?.run_id && loadRun(runStatus.run_id)}
                  disabled={!runStatus?.run_id}
                >
                  Refresh Status
                </button>
              </div>
              <label htmlFor='workflow-input'>Input JSON</label>
              <textarea id='workflow-input' value={inputText} onChange={(event) => setInputText(event.target.value)} />
              {selectedWorkflowAcceptsImages ? (
                <div className='workflow-image-input'>
                  <div>
                    <strong>Vision images</strong>
                    <span>
                      {workflowImages.length
                        ? `${workflowImages.length} image${workflowImages.length === 1 ? '' : 's'} attached`
                        : 'Attach image files for live vision workflow runs.'}
                    </span>
                  </div>
                  <label className='shovs-ghost-btn'>
                    Attach Images
                    <input
                      type='file'
                      accept='image/*'
                      multiple
                      onChange={(event) => {
                        if (event.target.files) addWorkflowImages(event.target.files);
                        event.currentTarget.value = '';
                      }}
                      style={{ display: 'none' }}
                    />
                  </label>
                  {workflowImages.length ? (
                    <div className='workflow-image-strip'>
                      {workflowImages.map((image) => (
                        <div key={image.id} className='workflow-image-chip'>
                          <img src={image.dataUrl} alt={image.name} />
                          <div>
                            <strong>{image.name}</strong>
                            <span>{formatBytes(image.size)}</span>
                          </div>
                          <button
                            type='button'
                            onClick={() => setWorkflowImages((prev) => prev.filter((item) => item.id !== image.id))}
                            aria-label={`Remove ${image.name}`}
                          >
                            Remove
                          </button>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>
              ) : null}
              <div className='workflow-schema-grid'>
                <div>
                  <span>Input schema</span>
                  <pre>{formatJson(selectedWorkflow.input_schema)}</pre>
                </div>
                <div>
                  <span>Output schema</span>
                  <pre>{formatJson(selectedWorkflow.output_schema)}</pre>
                </div>
              </div>
            </div>
          ) : null}

          {activeTab === 'events' ? (
            <div className='workflow-event-list'>
              {runStatus?.error ? <div className='workflow-error'>{runStatus.error}</div> : null}
              {events.length ? events.map((event) => (
                <div key={event.id} className={`workflow-event-row ${event.status}`}>
                  <span>{event.phase}</span>
                  <strong>{event.event_type}</strong>
                  <p>{event.summary}</p>
                </div>
              )) : <div className='workflow-empty'>Run the workflow to see status events.</div>}
            </div>
          ) : null}

          {activeTab === 'result' ? (
            <div className='workflow-result-panel'>
              {result ? (
                <>
                  <div className='workflow-result-head'>
                    <strong>{result.output_schema.type}</strong>
                    <span>{result.summary}</span>
                  </div>
                  <div className='workflow-result-grid'>
                    {Object.entries(result.fields).map(([field, value]) => (
                      <div key={field}>
                        <span>{field}</span>
                        <strong>{value.status}</strong>
                        <p>{value.value}</p>
                      </div>
                    ))}
                  </div>
                </>
              ) : <div className='workflow-empty'>No result yet.</div>}
            </div>
          ) : null}

          {activeTab === 'api' ? (
            <div className='workflow-api-panel'>
              <div className='workflow-api-actions'>
                <button className='shovs-ghost-btn' onClick={copyApiRequest}>
                  {copied ? 'Copied' : 'Copy API Request'}
                </button>
              </div>
              <pre>{formatJson(apiRequest)}</pre>
              <div className='workflow-legal-list'>
                {catalog.legal_usage.map((item) => <p key={item}>{item}</p>)}
              </div>
            </div>
          ) : null}
        </main>
      </div>
    </section>
  );
};
