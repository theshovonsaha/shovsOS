import React, { useState } from 'react';

interface GuardrailConfirmationModalProps {
  confirmation: {
    call_id: string;
    tool: string;
    arguments: Record<string, any>;
    preview: string;
    reason: string;
    created_at?: string;
  };
  onApprove: (callId: string) => void;
  onDeny: (callId: string, reason: string) => void;
}

export const GuardrailConfirmationModal: React.FC<
  GuardrailConfirmationModalProps
> = ({ confirmation, onApprove, onDeny }) => {
  const [denyReason, setDenyReason] = useState('');
  const [showDenyInput, setShowDenyInput] = useState(false);

  return (
    <aside className='guardrail-dock' aria-live='polite'>
      <div className='guardrail-dock-head'>
        <div>
          <div className='guardrail-dock-title'>Approval Waiting</div>
          <div className='guardrail-dock-subtitle'>{confirmation.tool}</div>
        </div>
        <span className='guardrail-dock-badge'>live</span>
      </div>

      <div className='guardrail-dock-preview'>{confirmation.preview}</div>

      {confirmation.reason ? (
        <div className='guardrail-dock-reason'>{confirmation.reason}</div>
      ) : null}

      <details className='guardrail-dock-args'>
        <summary>Arguments</summary>
        <pre>{JSON.stringify(confirmation.arguments, null, 2)}</pre>
      </details>

      {showDenyInput ? (
        <div className='guardrail-dock-deny'>
          <input
            value={denyReason}
            onChange={(e) => setDenyReason(e.target.value)}
            placeholder='reason for denial'
            autoFocus
          />
          <button
            className='guardrail-dock-btn danger'
            onClick={() =>
              onDeny(confirmation.call_id, denyReason.trim() || 'User denied')
            }
          >
            Confirm deny
          </button>
        </div>
      ) : null}

      <div className='guardrail-dock-actions'>
        <button
          className='guardrail-dock-btn subtle'
          onClick={() => setShowDenyInput((prev) => !prev)}
        >
          {showDenyInput ? 'Cancel deny' : 'Deny'}
        </button>
        <button
          className='guardrail-dock-btn approve'
          onClick={() => onApprove(confirmation.call_id)}
        >
          Approve
        </button>
      </div>
    </aside>
  );
};
