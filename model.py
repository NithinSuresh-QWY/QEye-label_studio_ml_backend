"""Image classifier ML backend for Label Studio (official SDK).

Wraps the fine-tuned EfficientNet-B0 checkpoint (``best_model.pth``) in a
:class:`label_studio_ml.model.LabelStudioMLBase` subclass for the standard
Label Studio ML backend server (``_wsgi.py`` / gunicorn).

On ``predict`` it loads each task image, runs inference, and returns predictions
in Label Studio's ``choices`` format. Images can come from S3 (``s3://...``,
downloaded directly with boto3) or from Label Studio uploaded / local / cloud
storage (downloaded via ``self.get_local_path`` which needs ``LABEL_STUDIO_URL``
and ``LABEL_STUDIO_API_KEY``).
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import boto3
import torch
import torch.nn as nn
import torch.nn.functional as F
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig
from PIL import Image
from torchvision import models, transforms

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_CLASSES = ["Bill", "Invalid", "Parcel", "Parcel_with_bill"]

MODEL_VERSION = os.getenv("MODEL_VERSION", "qeye-efficientnet-b0")


def _load_dotenv() -> None:
    """Load ``.env`` from this directory or the project root (if not already set)."""
    here = Path(__file__).resolve().parent
    for path in (here / ".env", here.parent / ".env"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()
# boto3 reads AWS_DEFAULT_REGION; our .env uses AWS_REGION.
os.environ.setdefault(
    "AWS_DEFAULT_REGION",
    os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-south-1",
)


def _resolve_model_path() -> str:
    candidates = [
        os.getenv("MODEL_PATH"),
        "best_model.pth",
        "../models/best_model.pth",
        os.path.join(os.getenv("MODEL_DIR", "/data/models"), "best_model.pth"),
        "/app/best_model.pth",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not locate best_model.pth. Set the MODEL_PATH env var to its location."
    )


def _build_model(num_classes: int) -> nn.Module:
    # weights=None: architecture only; real weights come from the checkpoint.
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    return model


class ImageClassifier(LabelStudioMLBase):
    """EfficientNet-B0 classifier. The torch model is loaded once per process."""

    _model = None
    _class_names: Optional[List[str]] = None
    _image_size = 224
    _device = None
    _transform = None
    _s3 = None

    def setup(self):
        """Lightweight init; checkpoint loads lazily on first predict."""
        self.set("model_version", MODEL_VERSION)

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._model is not None:
            return

        cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = _resolve_model_path()
        logger.info("Loading model checkpoint from %s", model_path)
        checkpoint = torch.load(model_path, map_location=cls._device, weights_only=False)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            cls._class_names = checkpoint.get("class_names", DEFAULT_CLASSES)
            cls._image_size = checkpoint.get("image_size", 224)
        else:
            state_dict = checkpoint
            cls._class_names = DEFAULT_CLASSES

        model = _build_model(len(cls._class_names))
        model.load_state_dict(state_dict)
        model.to(cls._device)
        model.eval()
        cls._model = model

        cls._transform = transforms.Compose(
            [
                transforms.Resize((cls._image_size, cls._image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")
        retries = {"max_attempts": 3, "mode": "standard"}
        no_sign = os.getenv("AWS_NO_SIGN_REQUEST", "").strip().lower() in {"1", "true", "yes"}
        boto_config = (
            BotoConfig(signature_version=UNSIGNED, retries=retries)
            if no_sign
            else BotoConfig(retries=retries)
        )
        cls._s3 = boto3.client("s3", region_name=region, config=boto_config)
        logger.info("Model loaded. Classes: %s", cls._class_names)

    def _load_image(self, task: Dict, image_uri: str) -> Image.Image:
        """Load a task image from S3 (s3://) or via the Label Studio helper."""
        if image_uri.startswith("s3://"):
            parsed = urlparse(image_uri)
            bucket, key = parsed.netloc, parsed.path.lstrip("/")
            obj = self._s3.get_object(Bucket=bucket, Key=key)
            data = obj["Body"].read()
            return Image.open(io.BytesIO(data)).convert("RGB")

        local_path = self.get_local_path(image_uri, task_id=task.get("id"))
        return Image.open(local_path).convert("RGB")

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """Run inference on a batch of tasks and return Label Studio predictions."""
        self._ensure_loaded()
        from_name, to_name, value = self.get_first_tag_occurence("Choices", "Image")

        predictions = []
        for task in tasks:
            image_uri = task["data"].get(value)
            try:
                image = self._load_image(task, image_uri)
            except Exception as exc:  # noqa: BLE001 - report but don't crash the batch
                logger.error("Failed to load %s: %s", image_uri, exc)
                predictions.append({"result": [], "score": 0.0})
                continue

            tensor = self._transform(image).unsqueeze(0).to(self._device)
            with torch.no_grad():
                probs = F.softmax(self._model(tensor), dim=1)[0]
                confidence, idx = torch.max(probs, dim=0)

            label = self._class_names[int(idx.item())]
            predictions.append(
                {
                    "model_version": self.get("model_version"),
                    "score": float(confidence.item()),
                    "result": [
                        {
                            "from_name": from_name,
                            "to_name": to_name,
                            "type": "choices",
                            "value": {"choices": [label]},
                        }
                    ],
                }
            )

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """Inference-only backend: training events are acknowledged and ignored."""
        logger.info("Received '%s' event; this backend is inference-only.", event)
