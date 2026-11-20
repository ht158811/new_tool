from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ProcessingScope(str, Enum):
    VALID_ANNOTATIONS = "valid_annotations"
    ALL_IMAGES = "all_images"
    WITHOUT_JSON = "without_json"


class OperationType(str, Enum):
    CROP = "crop"
    AUGMENTATION = "augmentation"
    LABEL_SPLIT = "label_split"
    LABEL_RENAME = "label_rename"
    CONVERSION = "conversion"


class ConversionTarget(str, Enum):
    YOLO = "yolo"
    COCO = "coco"
    VOC = "voc"


class LabelRetentionMode(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class CropConfig:
    rows: int = 2
    columns: int = 3
    keep_empty: bool = False


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    horizontal_flip: bool = False
    vertical_flip: bool = False
    rotate_90: bool = False
    rotate_180: bool = False
    rotate_270: bool = False
    brightness: float | None = None
    contrast: float | None = None
    saturation: float | None = None
    gaussian_blur: float | None = None
    gaussian_noise: float | None = None
    outputs_per_image: int = 1
    max_operations_per_output: int = 1
    random_seed: int = 42

    def enabled_operations(self) -> list[str]:
        operations: list[str] = []
        for enabled, name in (
            (self.horizontal_flip, "horizontal_flip"),
            (self.vertical_flip, "vertical_flip"),
            (self.rotate_90, "rotate_90"),
            (self.rotate_180, "rotate_180"),
            (self.rotate_270, "rotate_270"),
        ):
            if enabled:
                operations.append(name)
        for value, name in (
            (self.brightness, "brightness"),
            (self.contrast, "contrast"),
            (self.saturation, "saturation"),
            (self.gaussian_blur, "gaussian_blur"),
            (self.gaussian_noise, "gaussian_noise"),
        ):
            if value is not None:
                operations.append(name)
        return operations


@dataclass(frozen=True, slots=True)
class ConversionConfig:
    target: ConversionTarget = ConversionTarget.YOLO


@dataclass(frozen=True, slots=True)
class LabelSplitConfig:
    separator: str = ";"
    create_before: bool = True
    create_after: bool = True
    before_retention: LabelRetentionMode = LabelRetentionMode.BEFORE
    after_retention: LabelRetentionMode = LabelRetentionMode.AFTER
    before_group_by_label: bool = False
    after_group_by_label: bool = True

    def __post_init__(self) -> None:
        for name in ("before_retention", "after_retention"):
            value = getattr(self, name)
            if not isinstance(value, LabelRetentionMode):
                object.__setattr__(self, name, LabelRetentionMode(value))


@dataclass(frozen=True, slots=True)
class LabelRenameConfig:
    source_label: str
    target_label: str

    def __post_init__(self) -> None:
        source = self.source_label.strip()
        target = self.target_label.strip()
        if not source:
            raise ValueError("要修改的标签名不能为空")
        if not target:
            raise ValueError("修改后的标签名不能为空")
        if source == target:
            raise ValueError("修改前后的标签名不能相同")
        object.__setattr__(self, "source_label", source)
        object.__setattr__(self, "target_label", target)


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    project_id: str
    dataset_id: str
    output_root: str
    scope: ProcessingScope
    operation: OperationType
    result_name: str | None = None
    crop: CropConfig | None = None
    augmentation: AugmentationConfig | None = None
    label_split: LabelSplitConfig | None = None
    label_rename: LabelRenameConfig | None = None
    conversion: ConversionConfig | None = None

    def parameters(self) -> dict[str, Any]:
        selected = (
            self.crop or self.augmentation or self.label_split
            or self.label_rename or self.conversion
        )
        if selected is None:
            return {}
        data = asdict(selected)
        return {key: value.value if isinstance(value, Enum) else value for key, value in data.items()}


@dataclass(frozen=True, slots=True)
class ProcessingRecord:
    image_path: str
    annotation_path: str | None
    annotation_valid: bool
    issues: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    output_stem: str | None = None

    @property
    def has_json(self) -> bool:
        return self.annotation_path is not None


@dataclass(frozen=True, slots=True)
class ScopePreview:
    total_images: int
    valid_json: int
    without_json: int
    invalid_json: int
    selected_valid: int
    selected_without_json: int
    selected_invalid_json: int
    skipped: int

    @property
    def selected_total(self) -> int:
        return self.selected_valid + self.selected_without_json + self.selected_invalid_json


@dataclass(frozen=True, slots=True)
class ProcessingProgress:
    total: int
    processed: int
    succeeded: int
    failed: int
    skipped: int
    current_file: str
    status: str


@dataclass(slots=True)
class ProcessingResult:
    output_directory: str
    input_images: int
    selected_images: int
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    generated_images: int = 0
    generated_annotations: int = 0
    modified_shapes: int = 0
    warnings: list[str] = field(default_factory=list)
    cancelled: bool = False

