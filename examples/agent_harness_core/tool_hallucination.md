# Tool Hallucination

Problem: the model says a tool ran, but the runtime has no successful tool result.

Correct rule:

```text
No final answer may claim a tool result unless the run ledger contains a successful result ID.
```

The ledger stores:

- tool call ID
- tool name
- arguments
- result ID
- status
- summary

An orphan result is rejected. A failed result can still be shown, but the answer must describe it as failed.

