import React from 'react';

interface ToolInfo {
  name: string;
  description?: string;
}

interface AgentCapabilitiesViewProps {
  agentName: string;
  description?: string;
  allowedTools: ToolInfo[];
  globalTools: ToolInfo[];
}

export const AgentCapabilitiesView: React.FC<AgentCapabilitiesViewProps> = ({
  agentName,
  description,
  allowedTools,
  globalTools,
}) => {
  const globalOnly = globalTools.filter(
    (tool) => !allowedTools.some((allowed) => allowed.name === tool.name),
  );

  return (
    <section className='workspace-pane'>
      <div className='workspace-pane-head'>
        <div>
          <h2>Capabilities</h2>
          <p>
            Active agent behavior is shaped through conversation and controls.
            These tool boundaries define what it can actually execute.
          </p>
        </div>
      </div>

      <div className='workspace-grid two-up'>
        <div className='workspace-card emphasis'>
          <div className='workspace-card-kicker'>Active Agent</div>
          <div className='workspace-card-title'>{agentName}</div>
          <div className='workspace-card-copy'>
            {description?.trim() ||
              'This agent can be molded through chat. Capabilities, not preset modes, are the hard boundary.'}
          </div>
        </div>

        <div className='workspace-card'>
          <div className='workspace-card-kicker'>Execution Boundary</div>
          <div className='workspace-stat-row'>
            <span>Enabled tools</span>
            <strong>{allowedTools.length}</strong>
          </div>
          <div className='workspace-stat-row'>
            <span>Global registry</span>
            <strong>{globalTools.length}</strong>
          </div>
          <div className='workspace-card-copy subtle'>
            The system can know about more tools than this agent is allowed to
            use. That distinction is now explicit.
          </div>
        </div>
      </div>

      <div className='workspace-grid two-up'>
        <div className='workspace-card'>
          <div className='workspace-card-kicker'>Enabled Here</div>
          <div className='workspace-tool-list'>
            {allowedTools.length === 0 ? (
              <div className='workspace-empty-note'>
                No explicit tool grants on this agent.
              </div>
            ) : (
              allowedTools.map((tool) => (
                <div key={tool.name} className='workspace-tool-row enabled'>
                  <div className='workspace-tool-name'>{tool.name}</div>
                  <div className='workspace-tool-desc'>
                    {tool.description || 'No description available.'}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className='workspace-card'>
          <div className='workspace-card-kicker'>Available Globally</div>
          <div className='workspace-tool-list'>
            {globalOnly.length === 0 ? (
              <div className='workspace-empty-note'>
                This agent already has the full global tool set.
              </div>
            ) : (
              globalOnly.map((tool) => (
                <div key={tool.name} className='workspace-tool-row disabled'>
                  <div className='workspace-tool-name'>{tool.name}</div>
                  <div className='workspace-tool-desc'>
                    {tool.description || 'No description available.'}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </section>
  );
};
