# Sony Photo Sorter

面向 macOS 的索尼照片快速分拣工具。它把同名的 `.JPG` / `.JPEG` 与 `.ARW` 当作同一组，只显示一次；看片时只记录决定，直到用户确认“执行整理”后才将文件移入 macOS 系统废纸篓。

## 系统要求与启动

- macOS 14 或更新版本
- Xcode Command Line Tools（运行 `xcode-select --install` 安装）

最简单的启动方式：

```bash
cd ~/Code/stivine-tool/photo-sorter
./run.sh
```

也可以使用标准 Swift Package 命令：

```bash
swift run PhotoSorter
```

首次构建可能需要几十秒。脚本会生成并启动 `.build/PhotoSorter.app`，使用本机临时签名；它不会访问未由你选择的照片目录。

## 使用方法

1. 点击“选择文件夹”，选择 A6700 照片所在目录。只扫描当前层级，不进入子目录。
2. 浏览预览并按 `1`—`4` 分类。每次分类后自动前进；返回旧照片可覆盖原分类。
3. 分类状态和当前位置实时写入照片目录内的 `.photo-sorter.json`。关闭程序后重新选择同一目录即可恢复。
4. 点击“执行整理”查看最终摘要。未处理照片会明确列出并保持不动。
5. 确认后，待处理文件通过 `FileManager.trashItem` 移入系统废纸篓。失败不会中止其他文件，结束时会逐项显示原因。

右侧“拍摄参数”会异步读取照片元数据，显示相机、镜头、焦距、光圈、快门、ISO、曝光补偿、拍摄时间和分辨率。JPG 与 ARW 同时存在时优先读取 JPG；只有 ARW 时直接读取 RAW 元数据。缺失字段会自动隐藏，读取失败不会影响分类。

## 快捷键

| 按键 | 功能 |
| --- | --- |
| `1` | 废片：JPG 与 ARW 都移入废纸篓 |
| `2` | 全部保留 |
| `3` | JPG · 记录：保留 JPG，ARW 移入废纸篓 |
| `4` | RAW · 精修：保留 ARW，JPG 移入废纸篓 |
| `←` / `A` | 上一组 |
| `→` / `D` | 下一组 |
| `Space` / `Z` | 适应窗口 / 100% 切换 |
| `Cmd-Z` | 撤销上一次分类修改 |
| `Cmd-Shift-Z` | 重做 |
| `F` | 进入或退出全屏 |
| `S` | 添加或取消当前照片组的星标 |
| `+` / `-` | 放大或缩小预览（10%—800%） |
| `R` | 重新扫描目录，识别新生成的后期文件 |
| `V` | 在当前组的 JPG、RAW、精修图之间循环切换预览 |
| `Esc` | 退出 100% 状态；否则退出全屏 |

100% 模式下预览位于双向滚动区域，可拖动滚动条或使用触控板查看细节。只有 JPG 时“只留 RAW”不可用；只有 ARW 时“只留 JPG”不可用，快捷键误选也会给出提示。

预览区支持顶部 `-` / `+`、适应窗口、`1:1` 按钮和触控板捏合缩放。放大后可使用触控板或滚动条移动。右键照片可以复制预览图、复制文件、复制完整路径，或分别在 Finder 中显示 JPG、ARW 和后期文件。

顶部 `PS` 按钮会优先把 ARW 交给 Adobe Photoshop，此时 Photoshop 会按其设置进入 Camera Raw；没有 ARW 时则打开 JPG 或已识别的后期文件。工具会查找系统注册的 Photoshop，也兼容 `/Applications/Adobe Photoshop…/Adobe Photoshop….app` 目录结构。

星标与四种分类相互独立，不会改变整理计划。它会随分类一起实时保存，并在重新打开照片目录后恢复。

每组照片都可以在右侧“备注”区域记录挑选理由、后期思路或交付说明。内容实时写入状态文件，重新打开目录后恢复；清空备注不会影响分类。顶部“有备注”筛选可集中查看所有带备注的照片，底部显示备注总数。

顶部筛选菜单可以限定当前浏览范围：全部照片、未处理、星标、废片、全部保留、JPG · 记录、RAW · 精修、已后期、未后期。左右切换、自动前进、序号和相邻照片预载都会遵循当前筛选。例如选择“未处理”后，每次分类会自动跳到下一张未处理照片；选择“星标”后取消当前星标，会自动显示下一张星标照片。筛选选择也会保存到状态文件。

## 后期文件命名约定

推荐保留原始基本名，并在末尾添加以下任一后缀（不区分大小写）：

```text
DSC01234_edit.jpg
DSC01234-edited.tif
DSC01234_edited_1.jpg
DSC01234_final.jpeg
DSC01234_final_v2.jpeg
DSC01234-retouched.psd
DSC01234_精修.jpg
DSC01234-修图.tiff
```

支持后缀：`_edit`、`-edit`、`_edited`、`-edited`、`_final`、`-final`、`_retouched`、`-retouched`、`_精修`、`-精修`、`_修图`、`-修图`。这些后缀后还可以跟数字版本号，例如 `_edited_1`、`-edit-2`、`_final_v3`。`.tif`、`.tiff`、`.psd` 即使不带后缀，也会按同名基本名关联为后期文件。

从 Photoshop 保存或导出后，回到工具按 `R` 或点击顶部刷新按钮即可识别，无需丢失当前分类。后期文件始终受保护：四种分类的整理计划只处理原始 JPG/ARW，不会将后期成片移入废纸篓。

当一组同时包含原片和后期文件时，顶部会出现“预览版本”菜单，可以选择原片 JPG、原片 RAW 或任意精修图。底部始终显示当前实际预览的文件名；按 `V` 可循环切换。切换照片组后再返回时，本次运行中的预览选择仍会保留。

## 状态文件

`.photo-sorter.json` 是 UTF-8 JSON，示意如下：

```json
{
  "classifications" : {
    "dsc01234" : "keepRAW",
    "dsc01235" : "trash"
  },
  "currentGroupID" : "dsc01236",
  "selectedFilter" : "unprocessed",
  "starredGroupIDs" : [
    "dsc01234"
  ],
  "notes" : {
    "dsc01234" : "降低高光，保留肤色"
  },
  "updatedAt" : "2026-09-28T10:00:00Z",
  "version" : 1
}
```

键使用不区分大小写的基本文件名。状态文件损坏时程序会提示并以空状态继续；记录中已不存在的照片会被忽略。状态值为 `trash`、`keepAll`、`keepJPG`、`keepRAW`。

## 安全设计

- 分类阶段不移动照片。
- 执行前重新验证每个候选文件：必须位于所选目录当前层、必须属于本次扫描快照、文件仍须存在。
- 生产代码唯一的删除入口是 macOS `FileManager.trashItem(at:resultingItemURL:)`；没有使用永久删除 API。
- 系统负责处理废纸篓重名。
- 单个文件失败后继续执行；部分失败的照片组不会被当作完全成功，并可再次执行重试。

## 测试

```bash
swift test
```

如果本机 Command Line Tools 的 Swift Package Manager 与 SDK 版本不一致，也可运行不依赖包管理器的同等核心验证：

```bash
./validate.sh
```

测试全部使用系统临时目录和虚拟小文件，覆盖大小写配对、自然排序、单 JPG/ARW、四种计划、未处理保护、状态恢复、撤销/重做、扫描边界以及系统废纸篓 API。废纸篓集成测试只移动一个随机命名的单字节临时文件。

## 项目结构

- `PhotoScanner.swift`：非递归扫描、配对、自然排序
- `PhotoModels.swift` / `PhotoSession.swift`：分类模型、统计、撤销重做和浏览会话
- `StateStore.swift`：Codable JSON 原子保存和恢复
- `PreviewLoader.swift`：ImageIO JPG 缩略图、Quick Look RAW 预览与相邻 JPG 预载
- `OrganizationService.swift`：整理计划、安全复核、系统废纸篓执行
- `Views/`：SwiftUI 深色界面、确认摘要和失败报告

## 当前限制

- 当前 `.app` 使用本机临时签名，适合本机使用，尚未进行 Apple Developer ID 公证，不能直接作为公开发行安装包分发。
- ARW 预览依赖当前 macOS Quick Look 对相机 RAW 格式的支持；失败时显示占位信息，但分类仍可继续。
- 100% 模式支持触控板滚动查看，尚未实现连续手势缩放。
- 同一基本名同时存在多个 JPG 变体（例如 `.jpg` 和 `.jpeg`）时只选择自然排序靠前的一个配对，其他重复文件保持不动。
- 执行整理后撤销栈会随目录重新扫描而失去实际意义；已移入废纸篓的文件应通过 Finder 恢复，而不是使用分类撤销。
