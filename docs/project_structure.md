# 工程结构

```text
src/dataset_manager/
├── domain/                 # 纯模型、枚举、领域规则
├── application/            # 用例服务与抽象端口
├── infrastructure/         # JSON、路径、解析器等适配器
├── ui/                     # PySide6 窗口、页面、主题
├── main.py                 # 依赖装配与程序入口
└── __main__.py
tests/                      # 自动化测试与 fixtures
docs/                       # 需求和设计文档
```

运行时默认数据位于可执行程序旁的 `runtime/`：`config/`、`Projects/`、`Cache/`、`Temp/`、`Resources/`。开发时可通过 `DATASET_MANAGER_HOME` 指向其他绝对目录。

