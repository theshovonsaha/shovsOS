import React, { useState } from 'react';

interface PendingConfirmation {
  call_id: string;
  tool: string;
  arguments: Record<string, any>;
  preview: string;
  reason: string;
  created_at?: string;
}

interface OperatorInterventionsProps {
  sessionId?: string | null;
  isStreaming: boolean;
  pendingConfirmation?: PendingConfirmation | null;
  onApprove: (callId: string) => void;
  onDeny: (callId: string, reason: string) => void;
  onStop: () => void;
  variant?: 'full' | 'compact';
}

function formatCreatedAt(value?: string): string {
  if (!value) return 'pending now';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'pending now';
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

export const OperatorInterventions: React.FC<OperatorInterventionsProps> = ({
  sessionId,
  isStreaming,
  pendingConfirmation,
  onApprove,
  onDeny,
  onStop,
  variant = 'full',
}) => {
  const [denyReason, setDenyReason] = useState('');
  const [showDeny, setShowDeny] = useState(false);

  if (!sessionId && !pendingConfirmation && !isStreaming) {
    return null;
  }

  const hasPending = Boolean(pendingConfirmation);
  const shellClass =
    variant === 'compact'
      ? 'operator-controls operator-controls-compact'
      : 'operator-controls';

  return (
    <section className={shellClass}>
      <div className='operator-controls-head'>
        <div>
          <div className='operator-controls-title'>Operator Controls</div>
          <div className='operator-controls-subtitle'>
            {sessionId ? `session ${sessionId.slice(0, 10)}` : 'no active session'}
          </div>
        </div>
        <div className='operator-status-row'>
          <span className={`operator-status-pill ${isStreaming ? 'live' : 'idle'}`}>
            {isStreaming ? 'run active' : 'run idle'}
          </span>
          <span className={`operator-status-pill ${hasPending ? 'warn' : 'idle'}`}>
            {hasPending ? 'approval waiting' : 'no approvals pending'}
          </span>
        </div>
      </div>

      <div className='operator-action-row'>
        <button
          className='operator-btn'
          onClick={onStop}
          disabled={!isStreaming}
        >
          Stop run
        </button>
      </div>

      {pendingConfirmation ? (
        <div className='operator-approval-card'>
          <div className='operator-approval-top'>
            <div>
              <div className='operator-approval-title'>
                {pendingConfirmation.tool}
              </div>
              <div className='operator-approval-meta'>
                waiting since {formatCreatedAt(pendingConfirmation.created_at)}
              </div>
            </div>
            <span className='operator-approval-badge'>approval</span>
          </div>

          <div className='operator-approval-preview'>
            {pendingConfirmation.preview}
          </div>

          {pendingConfirmation.reason ? (
            <div className='operator-approval-reason'>
              {pendingConfirmation.reason}
            </div>
          ) : null}

          {variant === 'full' ? (
            <details className='operator-approval-args'>
              <summary>Inspect arguments</summary>
              <pre>{JSON.stringify(pendingConfirmation.arguments, null, 2)}</pre>
            </details>
          ) : null}

          {showDeny ? (
            <div className='operator-deny-row'>
              <input
                value={denyReason}
                onChange={(e) => setDenyReason(e.target.value)}
                placeholder='reason for denial'
              />
              <button
                className='operator-btn danger'
                onClick={() =>
                  onDeny(
                    pendingConfirmation.call_id,
                    denyReason.trim() || 'User denied',
                  )
                }
              >
                Confirm deny
              </button>
            </div>
          ) : null}

          <div className='operator-approval-actions'>
            <button
              className='operator-btn subtle'
              onClick={() => setShowDeny((prev) => !prev)}
            >
              {showDeny ? 'Cancel deny' : 'Deny'}
            </button>
            <button
              className='operator-btn approve'
              onClick={() => onApprove(pendingConfirmation.call_id)}
            >
              Approve
            </button>
          </div>
        </div>
      ) : null}
    </section>
  );
};
