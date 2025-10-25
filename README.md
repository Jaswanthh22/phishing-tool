# PhishGuard

PhishGuard is a reference project that demonstrates how to train a lightweight phishing email classifier, expose it
through a FastAPI service, and provide simple UI assets for analysts and awareness training.

## User value

- Jump-starts phishing detection with a reproducible TF-IDF + logistic regression pipeline and ready-made artifacts for deployment.
- Offers a FastAPI scoring service that plugs into mail gateways, SOAR workflows, or ad-hoc scripts via REST endpoints.
- Delivers analyst tooling—interactive dashboard, landing page, and sample templates—to support investigations and awareness exercises without extra frontend work.
- Includes curated starter data, automated reports, and pytest coverage so teams can retrain on their own corpora while tracking accuracy, precision, recall, and F1.
- Documents ethical guardrails and authorization controls to help organizations roll out the solution responsibly.

## Features

- **Training pipeline** – TF-IDF + logistic regression model trained on curated phishing and legitimate emails.
- **REST API** – FastAPI service for real-time scoring with health checks and JSON responses.
- **Interactive dashboard** – HTML/JS client that calls the API and visualizes predictions.
- **Starter dataset & templates** – Sample emails you can extend with your own intelligence.
- **Tests** – Pytest suite verifying the training pipeline stays functional.

## Project layout

```
├── api/                   # FastAPI service
├── data/                  # Training datasets
├── dashboard/             # Interactive prediction UI
├── landing/               # Static training overview page
├── ml/                    # Training scripts and artifacts
├── templates/             # Sample email templates
└── tests/                 # Pytest suite
```

## Getting started

### 1. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train the model

```bash
python ml/train_model.py
```

Artifacts written to `ml/`:

- `phishguard_model.joblib` – serialized pipeline
- `training_metrics.json` – accuracy/precision/recall/F1
- `classification_report.txt` – detailed class metrics
- `confusion_matrix.png` – quick visualization

Use `python ml/train_model.py --help` for additional options (custom dataset, output paths, etc.).

### 3. Run the API

```bash
uvicorn api.main:APP --reload
```

- `GET /health` – returns `{"status": "ready"}` once the model is loaded.
- `POST /predict` – send `{"text": "Your message"}` and receive a label plus confidence score.

### 4. Explore the UI

- `dashboard/index.html` – paste email content and call the API (requires the FastAPI server above).
- `landing/training.html` – static overview of the training workflow and dataset.

Serve these files with your preferred static file host (e.g., `python -m http.server`) or integrate them into your web
stack.

### 5. Run tests

```bash
pytest
```

The suite validates that the dataset loads and the training pipeline produces predictions.

## Customization tips

- Replace `data/emails.csv` with your labeled datasets. Keep the `text` and `label` columns; use the values
  `phishing` and `legitimate` for labels.
- Expand `ml/train_model.py` with new features (URL heuristics, sender metadata, etc.) and track metrics over time.
- Deploy the API behind authentication and integrate it with mail gateways, SOAR platforms, or custom tooling.
- Use the templates in `templates/` for awareness simulations or additional training data.

## License

Licensed under the MIT license. See `LICENSE` for details.
