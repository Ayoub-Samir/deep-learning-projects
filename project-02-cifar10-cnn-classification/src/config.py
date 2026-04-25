from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FEATURES_DIR = OUTPUTS_DIR / "features"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

SEED = 42
NUM_CLASSES = 10
VALIDATION_SIZE = 0.1
NUM_WORKERS = 0

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CUSTOM_IMAGE_SIZE = 32
PRETRAINED_IMAGE_SIZE = 224


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    family: str
    model_key: str
    loader_key: str
    batch_size: int
    epochs: int = 0
    optimizer: str | None = None
    learning_rate: float | None = None
    weight_decay: float = 0.0
    momentum: float = 0.9
    feature_reference: str | None = None
    svm_c: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


EXPERIMENT_SPECS = (
    ExperimentSpec(
        name="m1_lenet_like",
        family="torch",
        model_key="lenet_like",
        loader_key="custom",
        batch_size=128,
        epochs=10,
        optimizer="sgd",
        learning_rate=0.01,
        weight_decay=5e-4,
        momentum=0.9,
        notes="LeNet-5 inspired CNN written explicitly with PyTorch layers.",
    ),
    ExperimentSpec(
        name="m2_improved_cnn",
        family="torch",
        model_key="improved_cnn",
        loader_key="custom",
        batch_size=128,
        epochs=10,
        optimizer="sgd",
        learning_rate=0.01,
        weight_decay=5e-4,
        momentum=0.9,
        notes="Same LeNet-like hyperparameters with BatchNorm and Dropout improvements.",
    ),
    ExperimentSpec(
        name="m3_resnet18",
        family="torch",
        model_key="resnet18",
        loader_key="pretrained",
        batch_size=64,
        epochs=4,
        optimizer="sgd",
        learning_rate=0.005,
        weight_decay=1e-4,
        momentum=0.9,
        notes="Torchvision ResNet18 with ImageNet weights and partial fine-tuning.",
    ),
    ExperimentSpec(
        name="m4_resnet18_linear_svm_hybrid",
        family="hybrid",
        model_key="linear_svm_hybrid",
        loader_key="pretrained",
        batch_size=64,
        feature_reference="m3_resnet18",
        svm_c=1.0,
        notes="ResNet18 feature extractor followed by a Linear SVM on saved .npy features.",
    ),
)
