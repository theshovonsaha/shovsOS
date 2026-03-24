import React, { useEffect, useState, useRef } from 'react';
import { RichContentViewer } from './RichContentViewer';
import './ShovsView.css';

interface ShovsViewProps {
    onClose: () => void;
    isListening: boolean;
    isSpeaking: boolean;
    isThinking: boolean;
    lastUserText: string;
    currentAgentToken: string;
    lastAgentResponse: string;
    onToggleMic: () => void;
    voiceSensitivity: number;
    setVoiceSensitivity: (val: number) => void;
    voiceModel: string;
    setVoiceModel: (val: string) => void;
}

export const ShovsView: React.FC<ShovsViewProps> = ({
    onClose,
    isListening,
    isSpeaking,
    isThinking,
    lastUserText,
    currentAgentToken,
    lastAgentResponse,
    onToggleMic,
    voiceSensitivity,
    setVoiceSensitivity,
    voiceModel,
    setVoiceModel
}) => {
    const [flavor, setFlavor] = useState<'nebula' | 'ocean' | 'midnight'>('nebula');

    // Auto-scroll logic for the HUD display
    const hudRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
        if (hudRef.current) {
            hudRef.current.scrollTop = hudRef.current.scrollHeight;
        }
    }, [currentAgentToken, lastAgentResponse]);

    return (
        <div className={`shovs-container flavor-${flavor}`}>
            {/* Background Layers */}
            <div className="scenic-background">
                <div className="bg-glow" />
                <div className="bg-particles" />
            </div>

            {/* Top Navigation */}
            <nav className="shovs-nav">
                <button className="shovs-exit" onClick={onClose}>
                    <span className="exit-icon">✕</span>
                    <span className="exit-text">Exit shovs</span>
                </button>
                <div className="shovs-flavor-picker">
                    <button className={flavor === 'nebula' ? 'active' : ''} onClick={() => setFlavor('nebula')}>Nebula</button>
                    <button className={flavor === 'ocean' ? 'active' : ''} onClick={() => setFlavor('ocean')}>Ocean</button>
                    <button className={flavor === 'midnight' ? 'active' : ''} onClick={() => setFlavor('midnight')}>Midnight</button>
                </div>
                <div className="shovs-voice-settings">
                    <div className="setting-item">
                        <span className="setting-label">Sensitivity</span>
                        <input
                            type="range"
                            min="0" max="1" step="0.05"
                            value={voiceSensitivity}
                            onChange={(e) => setVoiceSensitivity(parseFloat(e.target.value))}
                        />
                    </div>
                    <div className="setting-item">
                        <span className="setting-label">Model</span>
                        <select value={voiceModel} onChange={(e) => setVoiceModel(e.target.value)}>
                            <option value="aura-orion-en">Aura Orion (Clean)</option>
                            <option value="aura-helios-en">Aura Helios (Warm)</option>
                            <option value="aura-hera-en">Aura Hera (Professional)</option>
                            <option value="edge:en-US-GuyNeural">Edge Guy (Fast)</option>
                            <option value="edge:en-GB-SoniaNeural">Edge Sonia (UK)</option>
                        </select>
                    </div>
                </div>
            </nav>

            {/* Centered Voice Core */}
            <div className={`voice-core-wrap ${isListening ? 'listening' : ''} ${isThinking ? 'thinking' : ''} ${isSpeaking ? 'speaking' : ''}`}>
                <div className="core-orb">
                    <div className="orb-inner" />
                    <div className="orb-ring r1" />
                    <div className="orb-ring r2" />
                    <div className="orb-pulse" />
                </div>
                <div className="core-label">
                    {isListening ? 'Listening...' : isThinking ? 'Thinking...' : isSpeaking ? 'Speaking...' : 'Standby'}
                </div>
            </div>

            {/* Live Text Overlays */}
            <div className="shovs-hud">
                {/* User Input Overlay (Ghost text) */}
                <div className={`user-stt-overlay ${isListening && lastUserText ? 'visible' : ''}`}>
                    <span className="stt-prefix">›</span> {lastUserText}
                </div>

                {/* Agent Response HUD */}
                <div className={`agent-response-hud ${isSpeaking || lastAgentResponse ? 'visible' : ''}`} ref={hudRef}>
                    <RichContentViewer content={lastAgentResponse + currentAgentToken} />
                </div>
            </div>

            {/* Bottom Mic Control */}
            <footer className="shovs-footer">
                <button
                    className={`shovs-mic-btn ${isListening ? 'active' : ''} ${isSpeaking ? 'processing' : ''}`}
                    onClick={onToggleMic}
                >
                    <div className="mic-wave" />
                    <span className="mic-icon">🎙</span>
                </button>
                <div className="shovs-hint">
                    {isListening ? 'Click to stop' : 'Click to speak'}
                </div>
            </footer>
        </div>
    );
};
