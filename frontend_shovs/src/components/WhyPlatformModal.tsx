import React from 'react';

interface Props {
    onClose: () => void;
}

interface Pillar {
    title: string;
    tagline: string;
    body: string;
    detail: string;
}

const PILLARS: Pillar[] = [
    {
        title: 'Voids & Updates',
        tagline: 'Memory that forgets correctly.',
        body: 'When you correct a fact ("actually I moved to Berlin"), the engine writes a void over the stale claim instead of leaving both versions in context. Updates are first-class — not an afterthought tacked onto a vector store.',
        detail: 'Subject/predicate keys deduplicate across durable + convergent streams. Corrections propagate without a manual purge.',
    },
    {
        title: 'Side-Effect Honesty',
        tagline: 'Tools tell the truth.',
        body: 'Every tool call returns a structured contract: what it changed, what it read, whether the runtime should treat it as observation or mutation. The planner sees real consequences, not a flat string blob.',
        detail: 'See engine/tool_contract.py — actions that modify state are flagged so verification phases can audit them.',
    },
    {
        title: 'Sticky Skills',
        tagline: 'Skills that activate when they should.',
        body: 'Skills declare triggers in their SKILL.md frontmatter. The loader keeps a registry; relevance is decided per-turn from the active goal set, not by hoping the model remembers a system-prompt instruction.',
        detail: '.agent/skills/{name}/SKILL.md — comma-separated triggers, eligibility hints, and per-skill bootstrap docs.',
    },
    {
        title: 'Phase-Aware Context',
        tagline: 'The right context at the right phase.',
        body: 'Memory items declare visibility per phase: PLANNING gets durable anchors, ACTING gets convergent task context, RESPONSE gets recent linear turns, VERIFICATION gets fact records. The packet shape changes with the phase.',
        detail: 'engine/context_schema.py defines ContextPhase; ContextItem carries phase_visibility frozensets.',
    },
    {
        title: 'Resonance',
        tagline: 'Coherence, not a fact list.',
        body: 'A second-pass scoring step lifts modules that share goals with confidently-relevant ones. The packet emerges as a coherent theme rather than a top-N grab bag — convergence with a small lift toward thematic agreement.',
        detail: 'DEFAULT_RESONANCE_WEIGHT=0.15 in context_engine_v2 — tunable per agent profile.',
    },
];

export const WhyPlatformModal: React.FC<Props> = ({ onClose }) => {
    return (
        <div
            className='modal-overlay'
            onClick={onClose}
            style={{
                position: 'fixed',
                inset: 0,
                background: 'rgba(8, 10, 18, 0.78)',
                zIndex: 1000,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '40px 20px',
                overflowY: 'auto',
            }}
        >
            <div
                onClick={(e) => e.stopPropagation()}
                style={{
                    maxWidth: '880px',
                    width: '100%',
                    background: 'var(--bg-panel, #14161f)',
                    border: '1px solid rgba(99,102,241,0.3)',
                    borderRadius: '14px',
                    padding: '32px 36px',
                    boxShadow: '0 20px 80px rgba(0,0,0,0.6)',
                    color: 'var(--text-primary, #e7e9f0)',
                }}
            >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                    <div>
                        <div style={{ fontSize: '11px', letterSpacing: '0.3em', opacity: 0.55, marginBottom: '6px' }}>
                            SHOVS // PLATFORM
                        </div>
                        <h2 style={{ margin: 0, fontSize: '22px', fontWeight: 600 }}>
                            Why this platform
                        </h2>
                    </div>
                    <button
                        onClick={onClose}
                        style={{
                            background: 'transparent',
                            border: '1px solid var(--border-color, rgba(255,255,255,0.1))',
                            color: 'inherit',
                            borderRadius: '6px',
                            padding: '4px 10px',
                            cursor: 'pointer',
                            fontSize: '13px',
                        }}
                    >
                        Close
                    </button>
                </div>
                <p style={{ marginTop: '4px', marginBottom: '24px', fontSize: '13px', opacity: 0.75, lineHeight: 1.55 }}>
                    A thinking runtime. Not a chat wrapper, not a RAG pipeline. Five
                    differentiators worth naming explicitly:
                </p>

                <div style={{ display: 'grid', gap: '14px' }}>
                    {PILLARS.map((p) => (
                        <div
                            key={p.title}
                            style={{
                                background: 'rgba(255,255,255,0.02)',
                                border: '1px solid rgba(255,255,255,0.06)',
                                borderRadius: '10px',
                                padding: '16px 18px',
                            }}
                        >
                            <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px', marginBottom: '6px' }}>
                                <div style={{ fontSize: '14px', fontWeight: 600 }}>{p.title}</div>
                                <div style={{ fontSize: '12px', opacity: 0.6 }}>{p.tagline}</div>
                            </div>
                            <div style={{ fontSize: '13px', lineHeight: 1.55, opacity: 0.88 }}>
                                {p.body}
                            </div>
                            <div style={{ marginTop: '8px', fontSize: '11px', opacity: 0.55, fontFamily: 'var(--mono, monospace)' }}>
                                {p.detail}
                            </div>
                        </div>
                    ))}
                </div>

                <div
                    style={{
                        marginTop: '24px',
                        padding: '14px 16px',
                        borderRadius: '10px',
                        background: 'linear-gradient(135deg, rgba(99,102,241,0.10), rgba(168,85,247,0.06))',
                        border: '1px solid rgba(99,102,241,0.25)',
                        fontSize: '12px',
                        lineHeight: 1.55,
                        opacity: 0.85,
                    }}
                >
                    <strong style={{ fontWeight: 600 }}>One context engine.</strong> v1/v2/v3
                    are gone as a user choice — the unified engine composes linear recency,
                    durable compression, convergent ranking, and resonance into a single
                    packet. Older sessions migrate transparently on first compress.
                </div>
            </div>
        </div>
    );
};
