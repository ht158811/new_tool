# Dataset Manager

面向 Windows 的本地离线视觉数据集管理工具。当前仓库已完成基础架构和 LabelMe 完整扫描闭环：四层模块、PySide6 主窗口、主题切换、项目管理、安全 JSON 存储、路径管理、LabelMe 只读解析器、后台扫描、取消和不可变扫描缓存。

在项目首页选择项目后，只需指定 LabelMe 数据集根目录。程序递归检索所有子文件夹，并将同一相对目录下同名的图片和 JSON 自动配对，例如 `line_a/001.jpg` 对应 `line_a/001.json`。扫描期间可取消，首次添加取消不会保存数据集，刷新取消会保留上一次完整结果。

“图片浏览”页面使用纯文件名列表，支持多选、文件名/类别筛选、完整图片适配与缩放、标注显隐，以及 LabelMe rectangle/polygon 叠加。列表不加载或显示缩略图。

图片默认完整适应预览区，可通过按钮或 Ctrl+滚轮缩放，放大后使用滚轮和滚动条浏览。文件与标注信息位于左侧文件列表下方，与列表同宽，并通过可拖动纵向分隔条调整高度；信息支持纵向滚动及 Shift+滚轮横向滚动。项目与数据集选择宽度也可通过分隔条调整。单击图片中的彩色标签名可修改该 shape 的类别名称；确认后使用原子写入同步到对应 LabelMe JSON，未知扩展字段保持不变。rectangle 同时兼容 LabelMe 标准双点和四角点写法。类别筛选索引需刷新数据集后更新。

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

