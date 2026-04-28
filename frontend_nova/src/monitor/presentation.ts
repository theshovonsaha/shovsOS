export type MonitorTone = 'neutral' | 'good' | 'warn';

export interface TraceEventSummary {
  id: string;
  ts: number;
  iso_ts?: string;
  agent_id?: string;
  session_id?: string;
  run_id?: string | null;
  event_type: string;
  pass_index?: number | null;
  size_bytes?: number;
  preview?: string;
  payload_ref?: string | null;
  data?: unknown;
}

export interface MonitorCard {
  id: string;
  title: string;
  eyebrow?: string;
  summary: string;
  detail?: string;
  tone: MonitorTone;
}

export interface TimelineEntry {
  id: string;
  stage: string;
  headline: string;
  lines: string[];
  passLabel?: string;
  toolName?: string;
  ts: number;
  tone: MonitorTone;
  runId?: string | null;
  eventType: string;
}

export interface PacketSection {
  title: string;
  body: string;
}

export interface RunReplaySummary {
  checkpoint_count: number;
  pass_count: number;
  artifact_count: number;
  eval_count: number;
  trace_event_count: number;
  evidence_count: number;
}

export interface RunReplayArtifact {
  artifact_id: string;
  artifact_type: string;
  label: string;
  tool_name?: string | null;
  size_bytes: number;
  preview?: string;
  created_at?: string;
}

export interface RunReplayCheckpoint {
  checkpoint_id: number;
  phase: string;
  tool_turn?: number;
  status?: string;
  strategy?: string;
  notes?: string;
  tools?: string[];
  created_at?: string;
}

export interface RunReplayPass {
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

export interface RunReplayEvidence {
  source: string;
  phase: string;
  tool_turn?: number | null;
  pass_id?: number | null;
  item_id: string;
  trace_id?: string | null;
  summary: string;
  provenance?: Record<string, unknown>;
}

export interface RunReplayResponse {
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

export interface ReplaySection {
  id: string;
  title: string;
  entries: MonitorCard[];
}

const HUMAN_EVENT_LABELS: Record<string, string> = {
  all: 'All Events',
  story: 'Readable Timeline',
  conversation_tension: 'Tension Check',
  stance_signals_extracted: 'Stance Signals',
  phase_context: 'Context Prepared',
  compiled_context: 'Message Prompt Ready',
  llm_pass_start: 'Model Started',
  llm_prompt: 'Prompt Sent',
  llm_pass_complete: 'Model Finished',
  tool_call: 'Tool Started',
  tool_result: 'Tool Finished',
  prompt_components: 'Prompt Built',
  assistant_response: 'Answer Ready',
  route_decision: 'Route Selected',
  verification_result: 'Verification Result',
  verification_warning: 'Verification Warning',
  manager_observation: 'Observation',
  plan: 'Planning',
};

export function humanizeTraceEvent(eventType: string): string {
  return HUMAN_EVENT_LABELS[eventType] || eventType.replace(/_/g, ' ');
}

function asObject(value: unknown): Record<string, any> {
  return value && typeof value === 'object' ? (value as Record<string, any>) : {};
}

function clipText(text: string, max = 180): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max - 3).trimEnd()}...`;
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

function latestEventOf(
  events: TraceEventSummary[],
  eventTypes: string[],
): TraceEventSummary | null {
  for (const event of events) {
    if (eventTypes.includes(event.event_type)) return event;
  }
  return null;
}

function humanStageLabel(eventType: string): string {
  switch (eventType) {
    case 'plan':
      return 'Planning';
    case 'tool_call':
      return 'Working';
    case 'tool_result':
    case 'manager_observation':
      return 'Checking Results';
    case 'verification_result':
    case 'verification_warning':
      return 'Verifying';
    case 'assistant_response':
      return 'Answering';
    case 'phase_context':
    case 'compiled_context':
      return 'Context Prepared';
    case 'conversation_tension':
    case 'stance_signals_extracted':
      return 'Tension Check';
    case 'route_decision':
      return 'Routing';
    default:
      return humanizeTraceEvent(eventType);
  }
}

export function describeTraceEvent(event: TraceEventSummary): string {
  const data = asObject(event.data);
  switch (event.event_type) {
    case 'runtime_loop_mode':
      return `Loop mode is ${data.effective || data.requested || 'unknown'}.`;
    case 'route_decision':
      return `The runtime chose the ${data.route_type || 'unknown'} route.`;
    case 'conversation_tension': {
      const summary = String(data.summary || event.preview || '').trim();
      const challenge = String(data.challenge_level || 'low').trim();
      const conflicts = Array.isArray(data.conflicting_facts)
        ? data.conflicting_facts.length
        : 0;
      const suffix = [
        challenge ? `challenge ${challenge}` : '',
        conflicts ? `${conflicts} conflict${conflicts === 1 ? '' : 's'}` : '',
      ]
        .filter(Boolean)
        .join(' · ');
      if (summary) return `${summary}${suffix ? ` · ${suffix}` : ''}`;
      return `A contradiction or stance drift was detected${suffix ? ` · ${suffix}` : ''}.`;
    }
    case 'stance_signals_extracted': {
      const count = Number(data.count || 0);
      const signals = Array.isArray(data.signals) ? data.signals : [];
      const preview = signals
        .slice(0, 2)
        .map((item) => String(item?.topic || item?.position || '').trim())
        .filter(Boolean)
        .join(', ');
      return `Detected ${count || signals.length} stance signal${count === 1 ? '' : 's'}${preview ? ` · ${preview}` : ''}.`;
    }
    case 'plan':
      return String(data.strategy || event.preview || 'The planner produced a strategy.');
    case 'phase_context':
    case 'compiled_context': {
      const { count, hasTension, preview } = summarizeIncludedItems(data);
      const phase = String(data.phase || 'current phase');
      const label =
        event.event_type === 'compiled_context'
          ? 'message prompt'
          : 'phase packet';
      const detail = [
        `${count} context item${count === 1 ? '' : 's'}`,
        hasTension ? 'tension visible' : '',
        preview ? `includes ${preview}` : '',
      ]
        .filter(Boolean)
        .join(' · ');
      return `${phase} ${label} prepared${detail ? ` · ${detail}` : ''}.`;
    }
    case 'tool_call':
      return `${data.tool_name || 'A tool'} started${data.arguments_summary ? ` · ${data.arguments_summary}` : '.'}`;
    case 'tool_result':
      return `${data.tool_name || 'A tool'} ${data.success === false ? 'failed' : 'returned'}${data.content_preview ? ` · ${data.content_preview}` : '.'}`;
    case 'manager_observation':
      return `${data.status || 'Observation complete'}${data.strategy ? ` · ${data.strategy}` : ''}`;
    case 'verification_result':
      return data.supported === false
        ? `Verification found issues${Array.isArray(data.issues) && data.issues.length ? ` · ${data.issues[0]}` : ''}.`
        : 'Verification passed.';
    case 'verification_warning':
      return Array.isArray(data.issues) && data.issues.length
        ? `Verification warning · ${data.issues[0]}`
        : 'Verification warning.';
    case 'assistant_response':
      return String(data.content || event.preview || 'The final response was saved.');
    default:
      return String(event.preview || humanizeTraceEvent(event.event_type));
  }
}

export function buildTimelineEntries(
  events: TraceEventSummary[],
  limit = 18,
): TimelineEntry[] {
  return events
    .filter((event) =>
      [
        'plan',
        'phase_context',
        'compiled_context',
        'tool_call',
        'tool_result',
        'manager_observation',
        'verification_result',
        'verification_warning',
        'assistant_response',
        'conversation_tension',
      ].includes(event.event_type),
    )
    .slice(0, limit)
    .map((event) => {
      const data = asObject(event.data);
      const stage = humanStageLabel(event.event_type);
      const passLabel =
        typeof event.pass_index === 'number' ? `Pass ${event.pass_index}` : undefined;
      const toolName = String(data.tool_name || '').trim() || undefined;
      const summary = describeTraceEvent(event);
      const lines = [summary];
      if (data.strategy && event.event_type !== 'plan') {
        lines.push(`Strategy: ${clipText(String(data.strategy), 140)}`);
      }
      if (data.notes && event.event_type !== 'manager_observation') {
        lines.push(`Notes: ${clipText(String(data.notes), 140)}`);
      }
      let tone: MonitorTone = 'neutral';
      if (
        event.event_type === 'verification_warning' ||
        (event.event_type === 'verification_result' && data.supported === false) ||
        (event.event_type === 'tool_result' && data.success === false) ||
        event.event_type === 'conversation_tension'
      ) {
        tone = 'warn';
      } else if (
        event.event_type === 'verification_result' ||
        (event.event_type === 'tool_result' && data.success !== false) ||
        event.event_type === 'assistant_response'
      ) {
        tone = 'good';
      }
      return {
        id: event.id,
        stage,
        headline: humanizeTraceEvent(event.event_type),
        lines: lines.slice(0, 2),
        passLabel,
        toolName,
        ts: event.ts,
        tone,
        runId: event.run_id,
        eventType: event.event_type,
      };
    });
}

export function buildMonitorOverview(args: {
  events: TraceEventSummary[];
  runReplay: RunReplayResponse | null;
  pendingConfirmation?: { tool?: string; preview?: string; reason?: string } | null;
}): { primary: MonitorCard[]; secondary: MonitorCard[] } {
  const { events, runReplay, pendingConfirmation } = args;
  const latestTool = latestEventOf(events, ['tool_result', 'tool_call']);
  const latestVerify = latestEventOf(events, ['verification_warning', 'verification_result']);
  const latestTension = latestEventOf(events, ['conversation_tension']);
  const latestResponse = latestEventOf(events, ['assistant_response']);
  const latestRoute = latestEventOf(events, ['route_decision', 'runtime_loop_mode']);

  const primary: MonitorCard[] = [];
  const secondary: MonitorCard[] = [];

  primary.push({
    id: 'objective',
    title: 'Current Objective',
    eyebrow: runReplay?.latest_pass?.phase || 'current run',
    summary: clipText(
      String(
        runReplay?.latest_pass?.objective ||
          runReplay?.latest_pass?.strategy ||
          latestResponse?.preview ||
          'Waiting for objective details.',
      ),
      220,
    ),
    tone: 'neutral',
  });

  primary.push({
    id: 'status',
    title: 'Current Status',
    eyebrow: runReplay?.run?.status || 'runtime',
    summary: clipText(
      String(
        runReplay?.latest_checkpoint?.strategy ||
          runReplay?.latest_checkpoint?.notes ||
          describeTraceEvent(latestRoute || latestVerify || latestTool || latestResponse || {
            id: 'empty',
            ts: 0,
            event_type: 'route_decision',
            preview: 'No runtime activity yet.',
          }),
      ),
      220,
    ),
    detail: runReplay?.run?.run_id ? `run ${runReplay.run.run_id.slice(0, 10)}` : undefined,
    tone:
      latestVerify?.event_type === 'verification_warning' ||
      asObject(latestVerify?.data).supported === false
        ? 'warn'
        : 'good',
  });

  if (latestTool) {
    primary.push({
      id: 'tool',
      title: 'Latest Tool Activity',
      eyebrow: humanStageLabel(latestTool.event_type),
      summary: clipText(describeTraceEvent(latestTool), 220),
      detail: typeof latestTool.pass_index === 'number' ? `pass ${latestTool.pass_index}` : undefined,
      tone:
        latestTool.event_type === 'tool_result' && asObject(latestTool.data).success === false
          ? 'warn'
          : 'neutral',
    });
  }

  if (latestVerify) {
    primary.push({
      id: 'verification',
      title: 'Verification State',
      eyebrow: humanizeTraceEvent(latestVerify.event_type),
      summary: clipText(describeTraceEvent(latestVerify), 220),
      tone:
        latestVerify.event_type === 'verification_warning' ||
        asObject(latestVerify.data).supported === false
          ? 'warn'
          : 'good',
    });
  }

  primary.push({
    id: 'evidence',
    title: 'Evidence Snapshot',
    eyebrow: `${runReplay?.summary?.evidence_count ?? 0} evidence item${(runReplay?.summary?.evidence_count ?? 0) === 1 ? '' : 's'}`,
    summary: clipText(
      String(
        runReplay?.evidence?.[0]?.summary ||
          'No evidence summary is available yet.',
      ),
      220,
    ),
    tone: (runReplay?.summary?.evidence_count || 0) > 0 ? 'good' : 'neutral',
  });

  primary.push({
    id: 'artifacts',
    title: 'Artifacts Snapshot',
    eyebrow: `${runReplay?.summary?.artifact_count ?? 0} artifact${(runReplay?.summary?.artifact_count ?? 0) === 1 ? '' : 's'}`,
    summary: clipText(
      String(
        runReplay?.artifacts?.[0]?.label ||
          runReplay?.artifacts?.[0]?.preview ||
          'No run artifacts have been stored yet.',
      ),
      220,
    ),
    tone: (runReplay?.summary?.artifact_count || 0) > 0 ? 'good' : 'neutral',
  });

  secondary.push({
    id: 'passes',
    title: 'Recent Passes',
    eyebrow: `${runReplay?.summary?.pass_count ?? 0} total`,
    summary: clipText(
      (runReplay?.passes || [])
        .slice(-3)
        .reverse()
        .map((pass) => `${humanStageLabel(pass.phase)} · ${pass.status || '--'}`)
        .join(' | ') || 'No pass records yet.',
      220,
    ),
    tone: 'neutral',
  });

  if (latestTension) {
    secondary.push({
      id: 'tension',
      title: 'Open Tension / Contradictions',
      eyebrow: 'attention',
      summary: clipText(describeTraceEvent(latestTension), 220),
      tone: 'warn',
    });
  }

  if (pendingConfirmation) {
    secondary.push({
      id: 'pending',
      title: 'Pending Intervention',
      eyebrow: pendingConfirmation.tool || 'approval needed',
      summary: clipText(
        pendingConfirmation.reason ||
          pendingConfirmation.preview ||
          'An operator approval is waiting.',
        220,
      ),
      tone: 'warn',
    });
  }

  return { primary, secondary };
}

export function buildReplaySections(
  runReplay: RunReplayResponse | null,
): ReplaySection[] {
  if (!runReplay) return [];
  const sections: ReplaySection[] = [];

  if ((runReplay.passes || []).length > 0) {
    sections.push({
      id: 'passes',
      title: 'Recent Passes',
      entries: (runReplay.passes || [])
        .slice(-4)
        .reverse()
        .map((pass) => ({
          id: `pass-${pass.pass_id}`,
          title: humanStageLabel(pass.phase),
          eyebrow: `pass ${pass.pass_id}`,
          summary: clipText(
            String(
              pass.objective ||
                pass.strategy ||
                pass.response_preview ||
                'No pass summary stored.',
            ),
            180,
          ),
          detail: pass.status || '--',
          tone: pass.status === 'verified' || pass.status === 'complete' ? 'good' : 'neutral',
        })),
    });
  }

  if ((runReplay.evidence || []).length > 0) {
    sections.push({
      id: 'evidence',
      title: 'Evidence Snapshot',
      entries: (runReplay.evidence || []).slice(0, 4).map((item, index) => ({
        id: `${item.trace_id || item.item_id}-${index}`,
        title: item.phase || 'phase',
        eyebrow: item.source,
        summary: clipText(String(item.summary || 'No evidence summary.'), 180),
        detail:
          typeof item.tool_turn === 'number' ? `turn ${item.tool_turn}` : undefined,
        tone: 'neutral',
      })),
    });
  }

  if ((runReplay.artifacts || []).length > 0) {
    sections.push({
      id: 'artifacts',
      title: 'Artifacts',
      entries: (runReplay.artifacts || []).slice(0, 4).map((artifact) => ({
        id: artifact.artifact_id,
        title: artifact.label,
        eyebrow: artifact.artifact_type,
        summary: clipText(
          String((artifact.preview || '').trim() || 'No preview stored.'),
          180,
        ),
        detail: artifact.tool_name || undefined,
        tone: 'good',
      })),
    });
  }

  return sections;
}

export function parsePacketSections(content: string): PacketSection[] {
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
