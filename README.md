# Dataset Manager

面向 Windows 的本地离线视觉数据集管理工具。当前仓库完成第一阶段基础架构：四层模块、PySide6 主窗口、主题切换、项目管理、安全 JSON 存储、路径管理，以及 LabelMe 数据模型和只读解析器。

## 环境

- Windows 10/11
- Python 3.10–3.14（界面使用精简的官方 PySide6 Essentials 组件）

## 安装与运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
dataset-manager
```

也可以运行：

```powershell
python -m dataset_manager
```

## 测试

```powershell
python -m unittest discover -s tests -v
```

详细设计见 `docs/`。

