# Security Policy

## Reporting

Please report security issues privately to the repository maintainers.

## Secret Handling

CodeGraph RAG should not require API keys for the default local embedding path.
Optional remote embedding providers read keys from environment variables only.

Never commit:

- `.env`
- API keys
- local SQLite indexes
- local benchmark datasets
- HuggingFace/model caches

## Local Code Privacy

The default embedding provider runs locally with
`jinaai/jina-embeddings-v2-base-code`. If you enable a remote embedding provider,
review your data policy because source snippets may be sent to that provider.
