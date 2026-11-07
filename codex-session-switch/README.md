# cswitch：独立 Codex 环境之间接力同一个会话

在独立的 `CODEX_HOME` 之间迁移并恢复同一个会话，保留会话 ID 和对话历史，使用目标环境自己的模型配置和登录凭据。

已在 macOS、Python 3.12、Codex CLI 0.166.0 上验证。此工具直接处理 Codex 本地存储格式，CLI 升级后需重新验证兼容性。

## 安装

依赖：Python 3.12、Codex CLI；读取压缩会话还需要 `zstd`。无需安装 Python 第三方包。

```zsh
git clone https://github.com/stivine/stivine-tool.git
cd stivine-tool/codex-session-switch

# 先确认依赖可用
python3.12 --version
codex --version
zstd --version

# 从当前目录建立启动入口；已有同名命令时 ln 会停止，避免覆盖
mkdir -p "$HOME/.local/bin"
ln -s "$PWD/cswitch.py" "$HOME/.local/bin/cswitch"
export PATH="$HOME/.local/bin:$PATH"

cswitch --help
```

将 `export PATH="$HOME/.local/bin:$PATH"` 加入自己的 shell 配置文件（如 `~/.zshrc`），即可在新终端直接使用。克隆目录需要保留，启动入口是指向脚本的符号链接。

工具不负责创建 Codex 环境或配置 API。先准备好各个目录中的 `config.toml` 和登录凭据；以下是默认的环境映射，启动别名仅为示例：

| 环境代号 | 配置与凭据目录 | 原有启动命令 |
| --- | --- | --- |
| `a` | `~/.codex` | `codex` |
| `b` | `~/.codex-b` | `cb` |
| `c` | `~/.codex-c` | `cnew` |

## 日常使用

先退出正在使用的那个会话，再在终端切换。其他无关会话可以继续运行；若所选会话引用的前序会话仍在使用，也会停止切换。

```zsh
# 在项目目录里执行，选择该项目的会话，交给 B 继续
cswitch b

# 选择会话，使用 C 自己配置的模型继续
cswitch c

# 交回原版继续
cswitch a

# 也可以指定 UUID 或唯一的 UUID 前缀
cswitch c 01a12345-6789-7abc-8def-0123456789ab

# 查看所有项目的会话和完整 ID
cswitch list --all

# 选择器查看其他项目的会话
cswitch b --all
```

工具自动查找各环境中的会话，并根据上次同步记录判断哪份有新增进度。切换时保留同一个会话 ID，复制完整记录及其前序依赖，使用目标环境的账号、供应商、模型与配置启动。每次覆盖前备份目标会话；不会复制 `auth.json`、密钥、配置、插件或整个数据库。

切换后的会话仍在目标环境保存。源环境保留上次的副本，因此之后继续接力请使用 `cswitch`，让工具同步最新历史；直接用 `codex resume`、`cb resume` 或 `cnew resume` 打开旧副本会形成不同进度。

没有填写默认模型的环境，会通过 Codex 自身的模型目录确定默认模型，避免继承来源环境的模型。目标账号是否登录、是否有模型权限，仍由该环境的 Codex 检查；`cswitch` 不会借用其他环境的凭据。

## 检查、冲突与备份

```zsh
cswitch c 会话ID --dry-run   # 检查来源、锁、冲突；不复制或启动
cswitch c 会话ID --no-launch # 只同步，不启动

# 两边都独立继续过，无法自动合并时，明确选择来源
cswitch c 会话ID --from b

# 目标也有独立修改时，明确允许备份后覆盖（不做历史合并）
cswitch c 会话ID --from b --replace
```

备份位于 `~/.local/share/cswitch/backups/<会话ID>/<时间>/`：

- `incoming/`：这次同步的来源会话包。
- `previous/`：覆盖前的目标会话包。
- `transaction.json`：同步是否完成；捕获到错误会恢复目标记录。

如果进程被强制终止、断电等，导致出现未完成事务，工具会停止后续自动切换并提示备份路径，需要检查备份后恢复，避免把未知状态当作最新历史。

## 导出与导入

导出内容包含对话、工具调用历史、压缩/分页历史依赖、会话元数据与目标状态；不包含账号配置和项目代码文件。

```zsh
cswitch export 会话ID /tmp/my-session --from b
cswitch import c /tmp/my-session

# 目标已有同 ID 的记录时，备份后覆盖
cswitch import c /tmp/my-session --replace

# 导入后继续
cswitch c 会话ID
```

会话包是一个目录，包含 `manifest.json` 和 `files/`，可自行压缩保存。历史文件有 SHA-256 校验。包中可能包含项目上下文和对话内容，请按项目资料保管。

## 范围与限制

- 针对同一台电脑上不同 `CODEX_HOME` 之间的串行接力，不支持两个环境同时写同一会话。
- 支持旧 JSONL、分页会话、压缩记录、fork/revert 引用的前序历史，保留正常的上下文压缩记录。
- 新版 Codex 会把部分历史写入 SQLite；工具只处理所选会话的行，不覆盖其他会话。数据库结构不同会停止迁移。
- 项目文件继续使用原项目路径，不会复制工作树、恢复 Git 状态或撤销文件修改。
- 不迁移运行中的终端进程、实时工具连接、子代理进程、环境技能/插件和个人记忆库。历史里的子代理记录保留在对话中，但不会迁移所有子会话。
- 排队消息未处理时拒绝切换，避免在另一环境意外执行。
- 文件/图片附件的历史和元数据会保留，但外部绝对路径的附件文件不额外打包；换电脑需要另外迁移项目及附件。
- 切换供应商后，历史上下文会发往目标环境配置的 API。
- 需要 Python 3.12；读取压缩会话还需 `zstd`。当前使用 POSIX 文件锁，不支持 Windows。

## 扩展环境

编辑 `~/.local/share/cswitch/config.json` 的 `homes` 映射即可增加更多独立 Codex 目录。每个目录必须有自己的配置与凭据。可用 `--settings /path/to/config.json` 临时指定映射；`CSWITCH_HOME` 环境变量可以隔离切换器自身的索引和备份。

源码位于本仓库的 `codex-session-switch/` 目录。日常使用无需修改现有 Codex 别名。

## 验证

```zsh
cd codex-session-switch
python3.12 -m unittest -v test_cswitch
```

集成测试运行真实 Codex CLI，模型接口为本机模拟服务，不消耗 API 额度。已验证 A → B → C → A 的上下文接续、旧格式、分页格式、压缩记录、fork/revert、正在使用的会话保护、分叉覆盖保护及导入导出。开发时另用 Kimi 接口对独立测试会话完成了真实接续与往返验证；该接口与凭据不包含在仓库内。
