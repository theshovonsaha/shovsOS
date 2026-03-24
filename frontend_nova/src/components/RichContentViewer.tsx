import React, { useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import DOMPurify from 'dompurify';
import 'katex/dist/katex.min.css';

interface RichContentViewerProps {
    content: string;
}

interface WebSearchResult {
    title: string;
    url: string;
    snippet: string;
}

interface StructuredPanelProps {
    title: string;
    subtitle?: string;
    summary: string;
    rawContent: string;
    children: React.ReactNode;
}

const FORBIDDEN_PREVIEW_ATTRIBUTES = ['onerror', 'onload', 'onclick', 'onmouseover', 'onfocus', 'onmouseenter', 'onmouseleave', 'onchange', 'onsubmit'];
const SAFE_HTML_PREVIEW_CONFIG = {
    USE_PROFILES: { html: true },
    FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'link', 'meta'],
    FORBID_ATTR: FORBIDDEN_PREVIEW_ATTRIBUTES,
};
const SAFE_SVG_PREVIEW_CONFIG = {
    USE_PROFILES: { svg: true, svgFilters: true },
    FORBID_TAGS: ['script', 'foreignObject'],
    FORBID_ATTR: FORBIDDEN_PREVIEW_ATTRIBUTES,
};

const buildPreviewDocument = (language: string, sanitizedMarkup: string) => {
    if (language === 'svg') {
        return [
            '<!doctype html>',
            '<html>',
            '<body style="margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:16px;background:#ffffff;">',
            sanitizedMarkup,
            '</body>',
            '</html>',
        ].join('');
    }

    return sanitizedMarkup;
};

const tryParseStructuredContent = (content: string) => {
    try {
        const trimmed = content.trim();
        const start = trimmed.indexOf('{');
        const end = trimmed.lastIndexOf('}');
        if (start !== -1 && end !== -1 && end > start) {
            return JSON.parse(trimmed.substring(start, end + 1));
        }
    } catch {
        // Fallback to markdown
    }

    return null;
};

const normalizeLLMMarkdown = (content: string) => {
    let normalized = content || '';

    normalized = normalized
        .replace(/<SYSTEM_TOOL_RESULT[\s\S]*?<\/SYSTEM_TOOL_RESULT>/gi, '')
        .replace(/<system-reminder>[\s\S]*?<\/system-reminder>/gi, '')
        .replace(/\[Tool Execution Turn\]/gi, '')
        .replace(/\bconfirmation_timeout\b/gi, '')
        .replace(/\r\n/g, '\n')
        .replace(/\t/g, '  ')
        .replace(/\n{3,}/g, '\n\n');

    // Normalize bullets from models that emit unicode/alt markers.
    normalized = normalized.replace(/^\s*[•▪◦▸▹►]\s+/gm, '- ');
    normalized = normalized.replace(/^(\s*)(\d+)\)\s+/gm, '$1$2. ');

    // Ensure headings and list sections are separated for markdown renderers.
    normalized = normalized.replace(/([^\n])\n(#{1,6}\s)/g, '$1\n\n$2');
    normalized = normalized.replace(/([^\n])\n(- |\d+\. )/g, '$1\n$2');

    return normalized.trim();
};

const StructuredPanel: React.FC<StructuredPanelProps> = ({
    title,
    subtitle,
    summary,
    rawContent,
    children,
}) => {
    const [expanded, setExpanded] = useState(false);
    const [showRaw, setShowRaw] = useState(false);

    return (
        <div className="structured-panel">
            <div className="structured-panel-head">
                <div className="structured-panel-title-wrap">
                    <div className="structured-panel-title">{title}</div>
                    {subtitle ? <div className="structured-panel-subtitle">{subtitle}</div> : null}
                </div>
                <div className="structured-panel-actions">
                    <button className="structured-panel-btn" onClick={() => setExpanded(prev => !prev)}>
                        {expanded ? 'Compact' : 'Expand'}
                    </button>
                    <button className="structured-panel-btn secondary" onClick={() => setShowRaw(prev => !prev)}>
                        {showRaw ? 'Hide Raw' : 'Show Raw'}
                    </button>
                </div>
            </div>

            <div className="structured-panel-summary">{summary}</div>

            {expanded ? <div className="structured-panel-body">{children}</div> : null}

            {showRaw ? (
                <details className="structured-panel-raw" open>
                    <summary>Raw payload sent back to the UI</summary>
                    <pre>{rawContent}</pre>
                </details>
            ) : null}
        </div>
    );
};

export const RichContentViewer: React.FC<RichContentViewerProps> = ({ content }) => {
    // Sanitize content to handle common LLM output issues that break KaTeX
    const sanitizedContent = normalizeLLMMarkdown(content)
        // Replace non-breaking hyphens with standard hyphens
        .replace(/\u2011/g, '-')
        // Replace smart/curly quotes with standard single/double quotes
        .replace(/[\u2018\u2019]/g, "'")
        .replace(/[\u201C\u201D]/g, '"')
        // Fix trailing % in math blocks that confuses KaTeX parser
        .replace(/%(\s*\$)/g, '$1');

    // Check if the content is a JSON result from tools
    const renderData = tryParseStructuredContent(sanitizedContent);

    if (renderData && renderData.type === 'web_search_results') {
        const results = Array.isArray(renderData.results) ? renderData.results as WebSearchResult[] : [];
        return (
            <StructuredPanel
                title="Web search sources"
                subtitle={renderData.query ? `Query: ${renderData.query}` : undefined}
                summary={`${results.length} result${results.length === 1 ? '' : 's'} collected`}
                rawContent={JSON.stringify(renderData, null, 2)}
            >
                <div className="search-results-viewer">
                    {results.map((r: WebSearchResult, i: number) => (
                        <div key={`${i}-${r.url}`} className="search-card">
                            <a href={r.url} target="_blank" rel="noreferrer" className="search-card-title">
                                {r.title}
                            </a>
                            <div className="search-card-url">{r.url}</div>
                            <div className="search-card-snippet">{r.snippet}</div>
                        </div>
                    ))}
                </div>
            </StructuredPanel>
        );
    }

    if (renderData && renderData.type === 'web_fetch_result') {
        const rawFetchContent = typeof renderData.content === 'string' ? renderData.content : '';
        return (
            <StructuredPanel
                title={renderData.error ? 'Page fetch failed' : 'Fetched page'}
                subtitle={renderData.url}
                summary={renderData.error
                    ? String(renderData.error)
                    : `${renderData.total_length || rawFetchContent.length || 0} chars ${renderData.truncated ? '(truncated)' : ''}`.trim()}
                rawContent={JSON.stringify(renderData, null, 2)}
            >
                <div className="fetch-result-viewer">
                    {renderData.error ? (
                        <div className="fetch-result-error">Error: {renderData.error}</div>
                    ) : (
                        <div className="fetch-result-content">
                            {rawFetchContent}
                            {renderData.truncated && (
                                <div className="fetch-result-note">
                                    Content truncated ({renderData.total_length} chars total)
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </StructuredPanel>
        );
    }

    if (renderData && renderData.type === 'markdown_preview' && typeof renderData.content === 'string') {
        return <RichContentViewer content={renderData.content} />;
    }

    if (
        renderData
        && (
            renderData.type === 'pdf_preview'
            || (typeof renderData.url === 'string' && renderData.url.toLowerCase().endsWith('.pdf'))
            || (typeof renderData.path === 'string' && renderData.path.toLowerCase().endsWith('.pdf'))
            || (typeof renderData.file === 'string' && renderData.file.toLowerCase().endsWith('.pdf'))
        )
    ) {
        const rawPdfPath = typeof renderData.url === 'string'
            ? renderData.url
            : (typeof renderData.path === 'string'
                ? renderData.path
                : (typeof renderData.file === 'string' ? `/sandbox/${renderData.file}` : ''));
        const pdfPath = rawPdfPath ? encodeURI(rawPdfPath) : '';
        if (!pdfPath) {
            return (
                <div className="pdf-render-sandbox rich-preview-sandbox" style={{ margin: '1em 0', borderRadius: '12px', border: '1px solid var(--border)', background: 'var(--surface2, #1e1e1e)', padding: '16px' }}>
                    <div style={{ fontWeight: 600, marginBottom: '8px' }}>PDF Preview unavailable</div>
                    <div style={{ fontSize: '13px', opacity: 0.8 }}>
                        The PDF preview path is missing from the tool response.
                    </div>
                </div>
            );
        }
        return (
            <div className="pdf-render-sandbox rich-preview-sandbox" style={{ margin: '1em 0', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border)', boxShadow: '0 8px 30px rgba(0,0,0,0.5)' }}>
                <div className="rich-preview-header" style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 20px', background: '#0a0a0a', borderBottom: '1px solid var(--border)' }}>
                    <span style={{ fontWeight: 700, fontSize: '13px', color: 'var(--primary)', letterSpacing: '0.02em', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '18px' }}>▣</span> {renderData.title || 'PDF PREVIEW'}
                    </span>
                    <a href={pdfPath} target="_blank" rel="noreferrer" style={{ fontSize: '11px', color: 'var(--text-dim)', textDecoration: 'none', background: 'rgba(255,255,255,0.05)', padding: '4px 10px', borderRadius: '4px' }}>OPEN PDF ↗</a>
                </div>
                <iframe
                    src={pdfPath}
                    title={renderData.title || 'PDF Preview'}
                    style={{ width: '100%', height: '720px', border: 'none', background: '#111' }}
                />
            </div>
        );
    }

    if (renderData && (renderData.path || renderData.type === 'app_view')) {
        const appPath = typeof renderData.path === 'string' ? encodeURI(renderData.path) : '';
        if (!appPath) {
            return (
                <div className="html-render-sandbox rich-preview-sandbox" style={{ margin: '1em 0', borderRadius: '12px', border: '1px solid var(--border)', background: 'var(--surface2, #1e1e1e)', padding: '16px' }}>
                    <div style={{ fontWeight: 600, marginBottom: '8px' }}>HTML Preview unavailable</div>
                    <div style={{ fontSize: '13px', opacity: 0.8 }}>
                        The app preview path is missing from the tool response.
                    </div>
                </div>
            );
        }
        return (
            <div className="html-render-sandbox rich-preview-sandbox" style={{ margin: '1em 0', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border)', boxShadow: '0 8px 30px rgba(0,0,0,0.5)' }}>
                <div className="rich-preview-header" style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 20px', background: '#0a0a0a', borderBottom: '1px solid var(--border)' }}>
                    <span style={{ fontWeight: 700, fontSize: '13px', color: 'var(--primary)', letterSpacing: '0.02em', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '18px' }}>◈</span> {renderData.title || 'V8 PLATINUM APP'}
                    </span>
                    <a href={appPath} target="_blank" rel="noreferrer" style={{ fontSize: '11px', color: 'var(--text-dim)', textDecoration: 'none', background: 'rgba(255,255,255,0.05)', padding: '4px 10px', borderRadius: '4px' }}>OPEN FULLSCREEN ↗</a>
                </div>
                <iframe
                    src={appPath}
                    title={renderData.title}
                    className="render-frame"
                    sandbox="allow-scripts allow-popups allow-same-origin"
                />
            </div>
        );
    }

    // Segment the content into thought and response blocks
    const segments = useMemo(() => {
        const parts: { type: 'thought' | 'content', text: string }[] = [];
        const regex = /<(THOUGHT|think)>([\s\S]*?)<\/\1>/gi;
        let lastIndex = 0;
        let match;

        while ((match = regex.exec(sanitizedContent)) !== null) {
            if (match.index > lastIndex) {
                parts.push({ type: 'content', text: sanitizedContent.substring(lastIndex, match.index) });
            }
            parts.push({ type: 'thought', text: match[2].trim() });
            lastIndex = regex.lastIndex;
        }
        
        if (lastIndex < sanitizedContent.length) {
            parts.push({ type: 'content', text: sanitizedContent.substring(lastIndex) });
        }
        
        return parts.length > 0 ? parts : [{ type: 'content', text: sanitizedContent }];
    }, [sanitizedContent]);

    return (
        <div className="rich-content-viewer">
            {segments.map((seg, idx) => (
                seg.type === 'thought' ? (
                    <ThoughtBlock key={idx} content={seg.text} />
                ) : (
                    <ReactMarkdown
                        key={idx}
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[
                            rehypeRaw,
                            [rehypeKatex, {
                                strict: 'ignore',
                                throwOnError: false,
                                trust: true
                            }]
                        ]}
                        components={{
                            code({ inline, className, children, ...props }: React.ComponentProps<'code'> & { inline?: boolean }) {
                                const match = /language-(\w+)/.exec(className || '');
                                const language = match ? match[1] : '';
                                const codeString = String(children).replace(/\n$/, '');

                                if (inline || !match) {
                                    return (
                                        <code className={className} {...props}>
                                            {children}
                                        </code>
                                    );
                                }

                                return <CodeBlock language={language} code={codeString} />;
                            },
                            table({ children, ...props }: React.ComponentProps<'table'>) {
                                return (
                                    <div style={{ overflowX: 'auto', margin: '1em 0' }}>
                                        <table {...props}>{children}</table>
                                    </div>
                                );
                            }
                        }}
                    >
                        {seg.text}
                    </ReactMarkdown>
                )
            ))}
        </div>
    );
};

const ThoughtBlock = ({ content }: { content: string }) => {
    const [expanded, setExpanded] = useState(false);

    return (
        <div className={`thought-block ${expanded ? 'expanded' : ''}`}>
            <div className="thought-header" onClick={() => setExpanded(!expanded)}>
                <span className="thought-icon">✧</span>
                <span className="thought-label">Thinking Process</span>
                <span className="thought-toggle">{expanded ? '−' : '+'}</span>
            </div>
            {expanded && (
                <div className="thought-body">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
                </div>
            )}
        </div>
    );
};

const CodeBlock = ({ language, code }: { language: string; code: string }) => {
    const [showPreview, setShowPreview] = useState(false);
    const [copied, setCopied] = useState(false);

    // Modular Live View logic: currently HTML and SVG are natively supported in the DOM.
    // Can be easily expanded to JSON visualizations, charts, etc.
    const isPreviewable = ['html', 'svg'].includes(language?.toLowerCase());
    const previewMarkup = useMemo(() => {
        if (!isPreviewable) return '';

        const previewLanguage = language?.toLowerCase();
        const sanitizedMarkup = DOMPurify.sanitize(
            code,
            previewLanguage === 'svg'
                ? SAFE_SVG_PREVIEW_CONFIG
                : SAFE_HTML_PREVIEW_CONFIG
        );

        return buildPreviewDocument(previewLanguage || '', sanitizedMarkup);
    }, [code, isPreviewable, language]);

    const copyToClipboard = () => {
        navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="code-block-container">
            <div className="code-block-header">
                <div className="code-block-language">
                    <span>{language}</span>
                    {isPreviewable && <span className="code-block-pill">preview available</span>}
                </div>
                <div className="code-block-actions">
                    {isPreviewable && (
                        <button
                            onClick={() => setShowPreview(!showPreview)}
                            className={`code-block-action ${showPreview ? 'active' : ''}`}
                        >
                            {showPreview ? 'CODE' : 'LIVE VIEW'}
                        </button>
                    )}
                    <button
                        onClick={copyToClipboard}
                        className={`code-block-action ${copied ? 'copied' : ''}`}
                    >
                        {copied ? 'COPIED!' : 'COPY'}
                    </button>
                </div>
            </div>

            {showPreview ? (
                <div className="code-preview-shell">
                    <div className="code-preview-note">Sandboxed live preview</div>
                    <iframe
                        className="code-preview-frame"
                        sandbox="allow-scripts"
                        srcDoc={previewMarkup}
                        title={`${language} live preview`}
                    />
                </div>
            ) : (
                <SyntaxHighlighter
                    style={vscDarkPlus}
                    language={language}
                    PreTag="div"
                    customStyle={{ margin: 0, borderRadius: 0, fontSize: '13px', background: '#0d0d0d' }}
                >
                    {code}
                </SyntaxHighlighter>
            )}
        </div>
    );
};
