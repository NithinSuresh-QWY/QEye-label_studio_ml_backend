"""Tests for the QEye ML backend API.

Run with the test requirements installed::

    pip install -r requirements-test.txt
    pytest

Image loading is monkeypatched so tests do not need S3 access.
"""

import json

import pytest
from PIL import Image

from model import ImageClassifier

LABEL_CONFIG = """
<View>
  <Image name="img" value="$image"/>
  <Choices name="choice" toName="img">
    <Choice value="Parcel"/>
    <Choice value="Bill"/>
    <Choice value="Parcel_with_bill"/>
    <Choice value="Invalid"/>
  </Choices>
</View>
"""


@pytest.fixture
def client():
    from label_studio_ml.api import init_app

    app = init_app(model_class=ImageClassifier)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert json.loads(response.data)["status"] == "UP"


def test_predict(client, monkeypatch):
    try:
        ImageClassifier._ensure_loaded()
    except FileNotFoundError:
        pytest.skip("best_model.pth not available; skipping inference test")

    monkeypatch.setattr(
        ImageClassifier,
        "_load_image",
        lambda self, task, image_uri: Image.new("RGB", (224, 224)),
    )

    request = {
        "tasks": [{"id": 1, "data": {"image": "s3://bucket/key.jpg"}}],
        "label_config": LABEL_CONFIG,
    }

    response = client.post(
        "/predict", data=json.dumps(request), content_type="application/json"
    )
    assert response.status_code == 200

    body = json.loads(response.data)
    result = body["results"][0]["result"][0]
    assert result["from_name"] == "choice"
    assert result["to_name"] == "img"
    assert result["type"] == "choices"
    assert result["value"]["choices"][0] in ImageClassifier._class_names
