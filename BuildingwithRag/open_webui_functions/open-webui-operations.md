# Open WebUI operations

## Reading answers (Story 3.2)

- `DRAFT — checking evidence` starts every attempt; the text below it is not yet validated.
- `Check failed: … Retrying (attempt 2 of 2)…` means that draft failed a check and a second attempt follows. Streamed text cannot be retracted, so earlier drafts stay visible.
- `Evidence check passed — confidence: high` + `Sources:` closes a validated answer.
- `DRAFT — low confidence, not the final answer.` means the final attempt failed; the reason and failed checks follow. Do not treat it as validated.
- Insufficient evidence is a single plain sentence; no confidence is shown.

## API key

Pipe Valve `capstone_api_key` must equal the server's `CAPSTONE_API_KEY` (both empty = no auth).

## Updating the Pipe

Open WebUI keeps its own copy of the Pipe: Function Menu → Edit, re-paste, save.
