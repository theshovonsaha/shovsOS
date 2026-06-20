# Small Model Guard

Problem: small or local models sometimes output tool JSON as a final answer.

Bad final answer:

```json
{"tool":"name_lookup","arguments":{"preferred_name":"Shovon"}}
```

Clean final answer:

```text
You should be called Shovon.
```

The response guard is not a replacement for tool validation. It is the final user-visible safety layer.

