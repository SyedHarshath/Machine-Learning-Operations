# Heart Disease ML + MySQL + Docker

A compact academic project that demonstrates:

- Heart-Disease dataset loaded through `sklearn.datasets.fetch_openml` and stored in MySQL
- Full CRUD operations for patient records
- Training a scikit-learn classifier using **only data fetched back from MySQL**
- Accuracy, precision, recall, F1 and ROC-AUC
- Confusion-matrix and ROC plots saved as binary images in MySQL
- A single interactive HTML dashboard
- Docker image + Docker Compose

## Architecture

`OpenML/scikit-learn -> Flask -> MySQL -> pandas -> scikit-learn -> MySQL plots/metrics -> HTML dashboard`

## Run

```bash
docker compose up --build -d
```

Open:

```text
http://localhost:5000
```

Check containers:

```bash
docker compose ps
docker compose logs -f web
```

Stop:

```bash
docker compose down
```

Delete database volume too:

```bash
docker compose down -v
```

## API

- `GET /api/data` - list rows
- `POST /api/data` - create row
- `PUT /api/data/<id>` - update row
- `DELETE /api/data/<id>` - delete row
- `POST /api/train` - train model
- `GET /api/metrics` - latest metrics
- `GET /api/plot/confusion_matrix` - latest confusion matrix image
- `GET /api/plot/roc_curve` - latest ROC image
- `GET /api/health` - health check

## MySQL tables

- `heart_records` - dataset + CRUD records
- `model_metrics` - training metrics and classification report
- `plots` - PNG images stored as `LONGBLOB`

## Important note about the dataset

The application uses the OpenML **Heart-Disease** dataset through scikit-learn's `fetch_openml`. It is commonly referenced as OpenML dataset 43398 and contains 303 rows and 13 predictors plus a target. The source values are converted to numeric values and the target is normalized to binary 0/1 for this demo.
