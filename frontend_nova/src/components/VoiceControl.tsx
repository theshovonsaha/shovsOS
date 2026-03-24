import React from 'react';

interface VoiceControlProps {
    isRecording: boolean;
    status: 'idle' | 'recording' | 'processing' | 'speaking';
    onToggle: () => void;
}

export const VoiceControl: React.FC<VoiceControlProps> = ({ isRecording, status, onToggle }) => {
    return (
        <div className="voice-control">
            <button
                className={`voice-btn ${status}`}
                onClick={onToggle}
                title={isRecording ? 'Stop Recording' : 'Start Voice Conversation'}
            >
                {status === 'idle' && <span>🎤</span>}
                {status === 'recording' && <span className="pulse">🔴</span>}
                {status === 'processing' && <span className="spin">⚙</span>}
                {status === 'speaking' && <span className="wave">🔊</span>}
            </button>
        </div>
    );
};
