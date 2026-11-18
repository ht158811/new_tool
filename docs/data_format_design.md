# 数据格式设计

## LabelMe

根对象保留 `version`、`flags`、`shapes`、`imagePath`、`imageData`、`imageHeight`、`imageWidth`。每个 Shape 保留 `label`、`points`、`group_id`、`shape_type`、`flags`；所有未知字段进入 `extra`，不得静默丢弃。

内部类别以名称为主，不在读取阶段生成或回写类别 ID。目标格式导出时才建立名称到 ID 的映射。

解析结果由文档和问题列表组成。问题至少覆盖 JSON 不合法、根结构错误、字段缺失/类型错误、空标注、异常类别名、点坐标非法或越界、不支持的 shape 类型、图片无法关联和声明尺寸不匹配。空标注使用提示级别，不一律判错。

rectangle 与 polygon 是首批可视化类型。其他 shape 完整保留，并产生明确的“不支持”问题。

## 任务规则

- 目标检测：只由明确支持的几何类型（首批 rectangle）生成检测对象。
- 语义分割：只由明确支持的 polygon 生成区域。
- 图像分类：需要独立配置图像级类别来源；不得默认将所有 `shape.label` 当作图片类别。

## 管理信息

每个项目目录包含 `project.json`，记录项目 ID、名称、时间、数据集配置和统计摘要。原始数据集只被绝对路径引用，不复制进管理目录。全局 `settings.json` 保存主题、管理根目录和默认版本目录。

缓存可重建，不作为事实来源。未来扫描缓存采用带 schema 版本的 JSONL，并通过临时文件原子提交。

