# 架构设计

## 分层

依赖方向为 `UI -> Application -> Domain`，Infrastructure 实现 Application 所需的持久化与解析能力，并依赖 Domain 数据模型。Domain 不依赖任何外层模块。

- **UI**：窗口、页面、对话框、主题和用户交互，只调用应用服务。
- **Application**：组织用例和事务边界。后续包含扫描、质量、文件、划分、合并、类别、转换、统计和版本服务。
- **Domain**：项目、数据集、图片、标注、Shape、类别、版本与划分等模型和规则。
- **Infrastructure**：原子 JSON 存储、路径、LabelMe/其他格式解析、图像读取和扫描缓存。

## 关键边界

`ProjectRepository` 与 `AnnotationParser` 使用抽象接口，使 JSON 实现和 LabelMe 实现可替换。解析结果携带结构化问题，不通过吞异常伪装成功。源 JSON 的未知字段保存在 `extra` 中，序列化时可以无损合并。

## 一致性设计

- 管理 JSON 写入同目录临时文件，刷新并 `fsync` 后使用 `os.replace` 原子提交。
- 完整扫描写临时缓存，仅成功结束时替换正式缓存；取消时丢弃临时数据。
- 文件批量操作将在应用层建立计划、预检冲突与空间，再逐项执行并记录可撤销步骤。
- 不确定的图片/标注匹配只报告问题，不执行联动修改。
- 标注类别编辑通过 Application 服务和 `AnnotationWriter` 端口执行；基础设施层只原子修改目标 `shapes[index].label`，UI 不直接写 JSON。

## 后台任务

扫描使用 `QThreadPool/QRunnable` 或专用 `QThread`。工作线程只产生不可变进度/结果消息；UI 线程只显示状态。任务管理器按数据集 ID 加锁，预留跨数据集并发策略。

当前实现使用专用 `QThread` 和合作式取消事件。同一数据集 ID 由 `ScanService` 互斥；不同数据集暂不设置全局并发上限。扫描结果以 `dataset_id/scan_id.json` 形式保存为不可变快照，项目配置只引用 `active_scan_id`。因此未完成或已取消扫描不会替换上次完整结果。

数据集显示名称不承担唯一键职责。不同 session 中允许存在同名相机目录，系统始终使用不可变数据集 ID 区分，并在界面显示/提示实际根路径。

