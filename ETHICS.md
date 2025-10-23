# Ethical considerations

Building phishing detection systems introduces both opportunities and responsibilities. To keep PhishGuard aligned with
responsible security practices:

1. **Respect privacy** – Only ingest emails that your organization is legally and contractually permitted to process.
   Redact or tokenize personal identifiers where possible and enforce strict access controls over the training data and
   model artifacts.

2. **Avoid overreach** – Supplement machine predictions with human review. Flag high-risk messages for analysts rather
   than automatically deleting them to reduce the chance of disrupting legitimate business communication.

3. **Account for bias** – Phishing content evolves constantly. Continuously retrain with diverse, current examples and
   audit for false positives that may disproportionately affect specific departments, vendors, or geographies.

4. **Transparent communication** – Inform employees that automated phishing detection is in place, explain how data is
   used, and offer opt-out channels where required by policy or regulation.

5. **Secure operations** – Treat the model, API, and collected telemetry as sensitive assets. Apply least privilege
   principles, monitor access, and ensure patches are applied promptly.

Following these guidelines helps ensure that machine learning augments security teams without eroding trust or privacy.
