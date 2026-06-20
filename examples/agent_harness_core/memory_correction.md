# Memory Correction

Problem: the user corrects a fact, but the agent keeps the old fact or loses both facts.

Example:

```text
My location is Vancouver.
Actually, I moved to Toronto.
```

Correct memory state:

| Fact | Status |
| --- | --- |
| User location is Toronto | current |
| User location was Vancouver | superseded |

Shovs memory uses temporal facts. Old facts are voided only as part of a replacement operation. If the new fact fails to store, the old fact remains current.

