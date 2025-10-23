# PhishGuard project proposal

## Objective

Create an end-to-end phishing detection reference implementation that covers data preparation, model training, API
deployment, and analyst-facing tooling. The solution should be lightweight enough for rapid experimentation while
illustrating patterns that scale to enterprise deployments.

## Scope

- Curate a labeled dataset of phishing and legitimate emails.
- Build a reproducible scikit-learn training pipeline with evaluation artifacts.
- Expose the trained model via a FastAPI microservice.
- Deliver simple web assets for awareness training and manual investigations.
- Document operational considerations, ethical guardrails, and authorization controls.

## Deliverables

1. `ml/train_model.py` pipeline with CLI options and generated metrics.
2. Serialized model artifacts ready for deployment.
3. FastAPI application supporting health checks and real-time predictions.
4. Dashboard and landing pages demonstrating analyst workflows.
5. Pytest suite to guard against regressions.
6. Documentation covering setup, usage, and governance.

## Success metrics

- End-to-end setup completes in under 30 minutes on a fresh workstation.
- Baseline model achieves >80% accuracy on the curated dataset (achieved in local tests).
- Prediction endpoint responds in under 200 ms for single-email requests.
- Tests execute successfully in CI environments.

## Timeline

The initial implementation is delivered in a single development sprint. Future work can focus on data expansion,
feature engineering, and integration with enterprise mail gateways.
