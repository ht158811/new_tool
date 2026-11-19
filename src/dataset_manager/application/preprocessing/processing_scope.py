from __future__ import annotations

from typing import Any

from dataset_manager.application.preprocessing.models import (
    ProcessingRecord, ProcessingScope, ScopePreview,
)


def records_from_cache(cache: dict[str, Any]) -> list[ProcessingRecord]:
    records: list[ProcessingRecord] = []
    raw_records = cache.get("records", [])
    if not isinstance(raw_records, list):
        raise ValueError("扫描缓存 records 字段损坏")
    for raw in raw_records:
        if not isinstance(raw, dict) or not isinstance(raw.get("image_path"), str):
            continue
        issues = raw.get("issues", [])
        annotation_valid = raw.get("annotation_valid")
        if not isinstance(annotation_valid, bool):
            annotation_valid = isinstance(raw.get("annotation_path"), str) and not any(
                isinstance(issue, dict) and issue.get("severity") == "error"
                for issue in issues if isinstance(issues, list)
            )
        records.append(ProcessingRecord(
            image_path=raw["image_path"],
            annotation_path=raw.get("annotation_path") if isinstance(raw.get("annotation_path"), str) else None,
            annotation_valid=annotation_valid,
            issues=tuple(issue for issue in issues if isinstance(issue, dict)) if isinstance(issues, list) else (),
        ))
    return records


def select_records(
    records: list[ProcessingRecord], scope: ProcessingScope
) -> list[ProcessingRecord]:
    if scope is ProcessingScope.VALID_ANNOTATIONS:
        return [record for record in records if record.annotation_valid]
    if scope is ProcessingScope.WITHOUT_JSON:
        return [record for record in records if not record.has_json]
    return list(records)


def build_preview(records: list[ProcessingRecord], scope: ProcessingScope) -> ScopePreview:
    valid = sum(record.annotation_valid for record in records)
    without = sum(not record.has_json for record in records)
    invalid = sum(record.has_json and not record.annotation_valid for record in records)
    selected = select_records(records, scope)
    selected_valid = sum(record.annotation_valid for record in selected)
    selected_without = sum(not record.has_json for record in selected)
    selected_invalid = sum(record.has_json and not record.annotation_valid for record in selected)
    return ScopePreview(
        total_images=len(records),
        valid_json=valid,
        without_json=without,
        invalid_json=invalid,
        selected_valid=selected_valid,
        selected_without_json=selected_without,
        selected_invalid_json=selected_invalid,
        skipped=len(records) - len(selected),
    )
