import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';

interface PremiumSelectProps {
    value: string;
    options: string[] | Record<string, string[]>;
    onChange: (value: string) => void;
    label?: string;
    placeholder?: string;
}

export const PremiumSelect: React.FC<PremiumSelectProps> = ({ value, options, onChange, label, placeholder }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [query, setQuery] = useState('');
    const containerRef = useRef<HTMLDivElement>(null);
    const triggerRef = useRef<HTMLDivElement>(null);
    const searchRef = useRef<HTMLInputElement>(null);
    const [dropdownStyle, setDropdownStyle] = useState<React.CSSProperties>({});

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (
                containerRef.current && 
                !containerRef.current.contains(event.target as Node) &&
                !(event.target as Element).closest('.premium-select-dropdown-portal')
            ) {
                setIsOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    useEffect(() => {
        if (isOpen) {
            window.setTimeout(() => searchRef.current?.focus(), 0);
        } else {
            setQuery('');
        }
    }, [isOpen]);

    useEffect(() => {
        if (!isOpen) return;

        const updatePosition = () => {
            const rect = triggerRef.current?.getBoundingClientRect();
            if (!rect) return;

            const viewportWidth = window.innerWidth;
            const desiredWidth = Math.min(Math.max(rect.width, 260), 360);
            const left = Math.max(12, Math.min(rect.right - desiredWidth, viewportWidth - desiredWidth - 12));
            const maxHeight = Math.max(180, Math.min(420, window.innerHeight - rect.bottom - 20));

            setDropdownStyle({
                position: 'fixed',
                top: rect.bottom + 8,
                left,
                width: desiredWidth,
                maxHeight,
                zIndex: 9999,
            });
        };

        updatePosition();
        window.addEventListener('resize', updatePosition);
        window.addEventListener('scroll', updatePosition, true);
        return () => {
            window.removeEventListener('resize', updatePosition);
            window.removeEventListener('scroll', updatePosition, true);
        };
    }, [isOpen]);

    const formatProviderLabel = (provider: string) => {
        const labels: Record<string, string> = {
            ollama: 'Ollama',
            lmstudio: 'LM Studio',
            llamacpp: 'llama.cpp',
            local_openai: 'Local OpenAI',
            openai: 'OpenAI',
            groq: 'Groq',
            gemini: 'Gemini',
            anthropic: 'Anthropic',
            nvidia: 'NVIDIA',
            all: 'All',
        };
        return labels[provider] || provider;
    };

    const handleSelect = (provider: string | null, model: string) => {
        if (provider) {
            onChange(`${provider}:${model}`);
        } else {
            onChange(model);
        }
        setIsOpen(false);
    };

    const getDisplayValue = () => {
        if (!value) return null;
        if (!value.includes(':')) return value;
        const [provider, model] = value.split(':', 2);
        return `${formatProviderLabel(provider)} / ${model}`;
    };

    const groupedOptions = Array.isArray(options)
        ? { 'all': options }
        : options;

    const normalizedQuery = query.trim().toLowerCase();
    const filteredOptions = Object.fromEntries(
        Object.entries(groupedOptions).map(([provider, models]) => [
            provider,
            (models || []).filter((model) => {
                if (!normalizedQuery) return true;
                const haystack = `${provider}:${model}`.toLowerCase();
                return haystack.includes(normalizedQuery) || model.toLowerCase().includes(normalizedQuery);
            }),
        ])
    );

    const hasOptions = Object.values(filteredOptions).some(list => list && list.length > 0);

    return (
        <div className="premium-select-container" ref={containerRef}>
            {label && <label className="premium-select-label">{label}</label>}
            <div
                ref={triggerRef}
                className={`premium-select-trigger ${isOpen ? 'active' : ''}`}
                onClick={() => setIsOpen(!isOpen)}
            >
                <div className="premium-select-value" title={value || placeholder || 'Select model'}>
                    {getDisplayValue() || <span className="placeholder">{placeholder || 'Select...'}</span>}
                </div>
                <div className="premium-select-arrow">▾</div>
            </div>

            {isOpen && createPortal(
                <div className="premium-select-dropdown premium-select-dropdown-portal" style={dropdownStyle}>
                    <div className="premium-select-search-wrap">
                        <input
                            ref={searchRef}
                            className="premium-select-search"
                            type="text"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="Search model or provider"
                            onClick={(e) => e.stopPropagation()}
                        />
                    </div>
                    {!hasOptions ? (
                        <div className="premium-select-no-options">
                            {normalizedQuery ? 'No matching models' : 'No options available'}
                        </div>
                    ) : (
                        Object.entries(filteredOptions).map(([provider, models]) => (
                            models && models.length > 0 && (
                                <div key={provider} className="provider-group">
                                    {provider !== 'all' && <div className="provider-header">{formatProviderLabel(provider)}</div>}
                                    {models.map((opt) => {
                                        const fullValue = provider !== 'all' ? `${provider}:${opt}` : opt;
                                        const isSelected = value === fullValue || value === opt;
                                        return (
                                            <div
                                                key={opt}
                                                className={`premium-select-option ${isSelected ? 'selected' : ''}`}
                                                onClick={() => handleSelect(provider === 'all' ? null : provider, opt)}
                                            >
                                                <span className="opt-text">{opt}</span>
                                                {isSelected && <span className="opt-check">✓</span>}
                                            </div>
                                        );
                                    })}
                                </div>
                            )
                        ))
                    )}
                </div>,
                document.body
            )}
        </div>
    );
};
