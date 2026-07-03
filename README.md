# QEye ML Backend (Label Studio)

EfficientNet-B0 image classifier exposed as a [Label Studio ML backend](https://labelstud.io/guide/ml_create)
using the official [`label-studio-ml`](https://github.com/HumanSignal/label-studio-ml-backend) SDK.

Classes (must match your trained checkpoint and Label Studio config): **Bill**, **Invalid**,
**Parcel**, **Parcel_with_bill**.

## Layout

```text
ml_backend/
  model.py                # ImageClassifier(LabelStudioMLBase)
  _wsgi.py                # SDK server entry point (gunicorn / flask)
  Dockerfile
  docker-compose.yml
  requirements-base.txt   # gunicorn + label-studio-ml SDK
  requirements.txt        # torch, torchvision, boto3, pillow
  requirements-test.txt
  test_api.py
  best_model.pth          # copy from ../models/ before building Docker image
```

## 1. Add the trained checkpoint

```bash
cp ../models/best_model.pth ./best_model.pth
```

## 2. Configure environment

For S3 task images (`s3://ls-order-images/...`), put AWS credentials in
`ml_backend/.env` (loaded automatically) or export them in your shell.
`docker-compose` also reads `ml_backend/.env` via `env_file`.

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=ap-south-1
```

For public buckets: `export AWS_NO_SIGN_REQUEST=1`

For non-`s3://` images (Label Studio uploads / local / cloud storage), also set:

```bash
export LABEL_STUDIO_URL=http://host.docker.internal:8080
export LABEL_STUDIO_API_KEY=<your-access-token>
```

## 3. Run with Docker (recommended)

```bash
docker-compose up --build
```

Validate:

```bash
curl http://localhost:9090/
# {"model_class":"ImageClassifier","status":"UP"}
```

## Run without Docker

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-base.txt -r requirements.txt
MODEL_PATH=../models/best_model.pth label-studio-ml start .
# or: MODEL_PATH=../models/best_model.pth python _wsgi.py --port 9090
```

## Test

```bash
pip install -r requirements-test.txt
MODEL_PATH=../models/best_model.pth pytest
```

## Connect to Label Studio

1. In your project: **Settings → Model → Add Model**.
2. Backend URL: `http://localhost:9090` (use `http://host.docker.internal:9090`
   if Label Studio runs in Docker on the same host).
3. Save — Label Studio calls `/setup` and the status turns green.

Labeling config (tag names must match `model.py`):

```xml
<View>
  <Image name="img" value="$image"/>
  <Choices name="choice" toName="img">
    <Choice value="Parcel"/>
    <Choice value="Bill"/>
    <Choice value="Parcel_with_bill"/>
    <Choice value="Invalid"/>
  </Choices>
</View>
```

## Auto-label unlabeled tasks

1. **Settings → Model → Retrieve predictions**
2. Data Manager → filter `Annotations = 0`
3. Review predictions → **Actions → Create Annotations From Predictions**

## Configuration

| Variable | Purpose |
|---|---|
| `MODEL_PATH` | Path to `best_model.pth` |
| `MODEL_VERSION` | Version string reported to Label Studio |
| `AWS_*` | S3 credentials / region |
| `AWS_NO_SIGN_REQUEST` | Read public S3 buckets without credentials |
| `LABEL_STUDIO_URL` / `LABEL_STUDIO_API_KEY` | Non-S3 image access |
| `ML_SERVER_BASIC_AUTH_USER` / `ML_SERVER_BASIC_AUTH_PASS` | Optional basic auth |
| `LOG_LEVEL`, `WORKERS`, `THREADS` | Server tuning |
