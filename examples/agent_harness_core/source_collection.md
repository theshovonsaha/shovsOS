# Source Collection

Problem: the agent finds the right entities, then searches for different entities later.

Example:

```text
Find the top 3 gainers.
For each ticker, search ticker-specific news.
Fetch exactly 3 URLs per ticker.
```

Correct workflow:

```mermaid
flowchart TD
    A["Fetch trusted mover page"] --> B["Lock entities"]
    B --> C["Search entity 1"]
    B --> D["Search entity 2"]
    B --> E["Search entity 3"]
    C --> F["Fetch 3 URLs"]
    D --> G["Fetch 3 URLs"]
    E --> H["Fetch 3 URLs"]
    F --> I["Summarize from fetched evidence"]
    G --> I
    H --> I
```

The source contract keeps the locked entities and URL quota visible to the runtime. The eval checks the trace, not just the final answer.

