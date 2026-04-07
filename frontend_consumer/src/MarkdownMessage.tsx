import { memo, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import type { Components } from 'react-markdown';

interface MarkdownMessageProps {
  content: string;
}

const components: Components = {
  // Open links in new tab
  a: ({ href, children, ...rest }) => (
    <a href={href} target='_blank' rel='noopener noreferrer' {...rest}>
      {children}
    </a>
  ),
  // Wrap code blocks with a copy button
  pre: ({ children, ...rest }) => {
    return (
      <div className='md-code-wrap'>
        <pre {...rest}>{children}</pre>
        <CopyButton
          getCode={() => {
            const el = document.createElement('div');
            // Extract text from React children
            if (rest.node?.children) {
              for (const child of rest.node.children) {
                if (child.type === 'element' && child.tagName === 'code') {
                  for (const textChild of child.children) {
                    if (textChild.type === 'text')
                      el.textContent += textChild.value;
                  }
                }
              }
            }
            return el.textContent || '';
          }}
        />
      </div>
    );
  },
};

function CopyButton({ getCode }: { getCode: () => string }) {
  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(getCode());
    } catch {
      /* fallback: noop */
    }
  }, [getCode]);

  return (
    <button
      className='md-copy-btn'
      onClick={copy}
      title='Copy code'
      aria-label='Copy code'
    >
      <svg
        width='12'
        height='12'
        viewBox='0 0 16 16'
        fill='none'
        stroke='currentColor'
        strokeWidth='1.5'
        strokeLinecap='round'
        strokeLinejoin='round'
      >
        <rect x='5' y='5' width='9' height='9' rx='1.5' />
        <path d='M5 11H3.5A1.5 1.5 0 0 1 2 9.5V3.5A1.5 1.5 0 0 1 3.5 2h6A1.5 1.5 0 0 1 11 3.5V5' />
      </svg>
    </button>
  );
}

export default memo(function MarkdownMessage({
  content,
}: MarkdownMessageProps) {
  return (
    <div className='md-root'>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});
