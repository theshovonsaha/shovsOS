import React, { useState } from 'react';

interface GuardrailConfirmationModalProps {
    confirmation: {
        call_id: string;
        tool: string;
        arguments: Record<string, any>;
        preview: string;
        reason: string;
    };
    onApprove: (callId: string) => void;
    onDeny: (callId: string, reason: string) => void;
}

export const GuardrailConfirmationModal: React.FC<GuardrailConfirmationModalProps> = ({
    confirmation,
    onApprove,
    onDeny
}) => {
    const [denyReason, setDenyReason] = useState('');
    const [showDenyInput, setShowDenyInput] = useState(false);

    return (
        <div className="modal-overlay" style={{ zIndex: 2000 }}>
            <div className="modal-content" style={{ maxWidth: '500px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '15px' }}>
                    <span style={{ fontSize: '24px', color: 'var(--primary)' }}>🛡️</span>
                    <h2 style={{ margin: 0 }}>Safety Confirmation</h2>
                </div>

                <div className="confirmation-details" style={{ background: 'rgba(255,255,255,0.05)', padding: '15px', borderRadius: '8px', marginBottom: '20px' }}>
                    <div style={{ fontSize: '12px', color: 'var(--text-dim)', marginBottom: '5px', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                        Tool Execution Requested
                    </div>
                    <div style={{ fontSize: '18px', fontWeight: 600, color: 'var(--primary)', marginBottom: '10px' }}>
                        {confirmation.tool}
                    </div>
                    <div style={{ fontSize: '14px', lineHeight: '1.5', color: 'var(--text)' }}>
                        {confirmation.preview}
                    </div>

                    {confirmation.reason && (
                        <div style={{ marginTop: '15px', fontSize: '12px', color: '#ffb300', background: 'rgba(255, 179, 0, 0.1)', padding: '8px', borderRadius: '4px', borderLeft: '3px solid #ffb300' }}>
                            <strong>Reason:</strong> {confirmation.reason}
                        </div>
                    )}
                </div>

                <div className="args-preview" style={{ marginBottom: '20px' }}>
                    <label style={{ fontSize: '10px', color: 'var(--text-dim)', textTransform: 'uppercase', display: 'block', marginBottom: '8px' }}>Arguments</label>
                    <pre style={{
                        fontSize: '11px',
                        background: '#000',
                        padding: '10px',
                        borderRadius: '6px',
                        overflow: 'auto',
                        maxHeight: '150px',
                        border: '1px solid var(--border)'
                    }}>
                        {JSON.stringify(confirmation.arguments, null, 2)}
                    </pre>
                </div>

                {showDenyInput && (
                    <div style={{ marginBottom: '20px' }}>
                        <label style={{ fontSize: '10px', color: 'var(--text-dim)', textTransform: 'uppercase', display: 'block', marginBottom: '8px' }}>Denial Reason</label>
                        <input
                            value={denyReason}
                            onChange={e => setDenyReason(e.target.value)}
                            placeholder="e.g. security concern, incorrect parameters..."
                            autoFocus
                            style={{
                                width: '100%',
                                padding: '10px',
                                borderRadius: '6px',
                                border: '1px solid var(--border)',
                                background: '#111',
                                color: 'var(--text)',
                                outline: 'none'
                            }}
                        />
                    </div>
                )}

                <div className="modal-actions" style={{ justifyContent: 'space-between' }}>
                    {!showDenyInput ? (
                        <button
                            className="danger-btn"
                            onClick={() => setShowDenyInput(true)}
                            style={{ background: 'transparent', border: '1px solid var(--error)', color: 'var(--error)' }}
                        >
                            Deny Request
                        </button>
                    ) : (
                        <button
                            className="danger-btn"
                            onClick={() => onDeny(confirmation.call_id, denyReason || 'User denied')}
                        >
                            Confirm Denial
                        </button>
                    )}

                    <button
                        className="primary-btn"
                        onClick={() => onApprove(confirmation.call_id)}
                        style={{ padding: '10px 24px', fontWeight: 600 }}
                    >
                        Approve execution
                    </button>
                </div>
            </div>
        </div>
    );
};
