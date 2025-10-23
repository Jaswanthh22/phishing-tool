# Authorization plan

Implement the following controls before deploying PhishGuard in production:

## Data access

- Restrict raw email datasets and model artifacts to a dedicated security analytics storage account.
- Require multi-factor authentication for all analysts with read/write access to the repository or API host.
- Log every dataset download or model export for audit purposes.

## API security

- Place the FastAPI service behind a reverse proxy that enforces organization single sign-on (SSO).
- Issue scoped API tokens for automation. Rotate tokens quarterly and immediately on personnel changes.
- Rate-limit prediction requests per user to reduce abuse or brute-force attempts.

## Infrastructure

- Run the service on hardened hosts with automatic security updates enabled.
- Store secrets (API keys, database credentials) in a managed secrets vault rather than environment variables baked into code.
- Continuously monitor logs for anomalous access patterns and integrate alerts with the security operations center.

By coupling the detection model with these controls, the system stays aligned with corporate security and compliance requirements.
