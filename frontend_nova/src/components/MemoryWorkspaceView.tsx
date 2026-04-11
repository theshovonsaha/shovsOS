import React from 'react';

interface SessionMemoryState {
  summary: {
    deterministic_fact_count: number;
    superseded_fact_count: number;
    candidate_signal_count: number;
    stance_signal_count: number;
    context_line_count: number;
    memory_signal_count: number;
  };
  deterministic_facts: Array<{
    subject: string;
    predicate: string;
    object: string;
    created_at?: string | null;
  }>;
  candidate_signals: Array<{
    text: string;
    reason: string;
    confidence?: string;
    topic?: string;
  }>;
  recent_memory_signals: Array<{
    event_type: string;
    label: string;
    summary: string;
    created_at?: string;
  }>;
  explanation: string[];
}

interface MemoryWorkspaceViewProps {
  sessionId?: string | null;
  memoryState: SessionMemoryState | null;
}

export const MemoryWorkspaceView: React.FC<MemoryWorkspaceViewProps> = ({
  sessionId,
  memoryState,
}) => {
  if (!sessionId) {
    return (
      <section className='workspace-pane'>
        <div className='workspace-pane-head'>
          <div>
            <h2>Memory</h2>
            <p>Start or load a session to inspect active memory state.</p>
          </div>
        </div>
      </section>
    );
  }

  if (!memoryState) {
    return (
      <section className='workspace-pane'>
        <div className='workspace-pane-head'>
          <div>
            <h2>Memory</h2>
            <p>Loading session memory state...</p>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className='workspace-pane'>
      <div className='workspace-pane-head'>
        <div>
          <h2>Memory</h2>
          <p>
            Trusted facts, candidate signals, and recent memory decisions for
            the active session.
          </p>
        </div>
      </div>

      <div className='workspace-grid metrics'>
        <div className='workspace-card metric'>
          <div className='workspace-card-kicker'>Trusted facts</div>
          <div className='workspace-metric'>
            {memoryState.summary.deterministic_fact_count}
          </div>
        </div>
        <div className='workspace-card metric'>
          <div className='workspace-card-kicker'>Candidate signals</div>
          <div className='workspace-metric'>
            {memoryState.summary.candidate_signal_count}
          </div>
        </div>
        <div className='workspace-card metric'>
          <div className='workspace-card-kicker'>Stance signals</div>
          <div className='workspace-metric'>
            {memoryState.summary.stance_signal_count}
          </div>
        </div>
        <div className='workspace-card metric'>
          <div className='workspace-card-kicker'>Memory events</div>
          <div className='workspace-metric'>
            {memoryState.summary.memory_signal_count}
          </div>
        </div>
      </div>

      <div className='workspace-grid two-up'>
        <div className='workspace-card'>
          <div className='workspace-card-kicker'>Deterministic Facts</div>
          <div className='workspace-fact-list'>
            {memoryState.deterministic_facts.length === 0 ? (
              <div className='workspace-empty-note'>No trusted facts yet.</div>
            ) : (
              memoryState.deterministic_facts.slice(0, 12).map((fact, index) => (
                <div key={`${fact.subject}-${fact.predicate}-${index}`} className='workspace-fact-row'>
                  <span>{fact.subject}</span>
                  <span>{fact.predicate}</span>
                  <strong>{fact.object}</strong>
                </div>
              ))
            )}
          </div>
        </div>

        <div className='workspace-card'>
          <div className='workspace-card-kicker'>Candidate Signals</div>
          <div className='workspace-tool-list'>
            {memoryState.candidate_signals.length === 0 ? (
              <div className='workspace-empty-note'>
                No candidate signals held for review.
              </div>
            ) : (
              memoryState.candidate_signals.slice(0, 10).map((signal, index) => (
                <div key={`${signal.text}-${index}`} className='workspace-tool-row disabled'>
                  <div className='workspace-tool-name'>{signal.text}</div>
                  <div className='workspace-tool-desc'>
                    {signal.reason}
                    {signal.topic ? ` · ${signal.topic}` : ''}
                    {signal.confidence ? ` · ${signal.confidence}` : ''}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className='workspace-grid two-up'>
        <div className='workspace-card'>
          <div className='workspace-card-kicker'>Recent Memory Signals</div>
          <div className='workspace-tool-list'>
            {memoryState.recent_memory_signals.length === 0 ? (
              <div className='workspace-empty-note'>
                No recent memory write events.
              </div>
            ) : (
              memoryState.recent_memory_signals.slice(0, 8).map((signal, index) => (
                <div key={`${signal.event_type}-${index}`} className='workspace-tool-row enabled'>
                  <div className='workspace-tool-name'>{signal.label}</div>
                  <div className='workspace-tool-desc'>{signal.summary}</div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className='workspace-card'>
          <div className='workspace-card-kicker'>Policy Notes</div>
          <div className='workspace-note-list'>
            {memoryState.explanation.map((line, index) => (
              <div key={`${line}-${index}`} className='workspace-note-line'>
                {line}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};
