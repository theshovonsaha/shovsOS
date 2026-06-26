# Thesis

The model is the language engine. The harness is the physics.

Prompts tell the model what to try. The harness decides what is valid, what is missing, what can be cited, and whether the run is allowed to finish.

## The Minimal Shape

```text
request
  -> contract
  -> ledger
  -> action
  -> observation
  -> attention
  -> verification
  -> response or continuation
```

Each part is independently testable.

## The Practical Bet

Enterprise users do not need a new chatbot first. They need agent runs that can answer:

- What was the task?
- What tools were allowed?
- Which tool calls actually happened?
- Which outputs came back?
- Which claims are supported by which result IDs?
- Why did the agent stop?
- What is still missing?

This folder is the smallest version of that bet.
