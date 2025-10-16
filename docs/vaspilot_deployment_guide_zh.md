# VASPilot 部署与扩展指南

本指南面向希望在实验室或生产环境中部署 **VASPilot**、以及二次开发新 Agent 和工具的使用者。我们将从环境准备、安装部署、服务启动、典型使用流程讲起，最后详细说明如何扩展体系。

---

## 1. 项目概览

VASPilot 基于 CrewAI 多智能体框架与 Model Context Protocol (MCP) 构建，核心组件如下：

- **Quart/Flask Web 服务**：提供任务提交流程与状态面板，对外暴露 REST API。
- **CrewAI 智能体集群**：经理 Agent 调度晶体结构、VASP 计算、结果验证等专业 Agent。
- **MCP 服务器**：封装 VASP 计算、结构搜索、绘图等能力，供 Agent 调用。
- **存储系统**：以 SQLite 记录计算元数据，同时通过工作目录保留输入输出。

项目源码关键位置：

- `src/vaspilot/server/`：Web 服务启动入口。
- `src/vaspilot/crew/vasp_crew.py`：CrewAI Agent 构建逻辑。
- `src/vaspilot/tools/`：本地工具与 MCP 工具实现。
- `examples/`：完整的最小可运行配置示例。

---

## 2. 环境准备

### 2.1 系统要求

- Linux (推荐) 或其他带有 Slurm 环境的 POSIX 系统。
- Python 3.10 及以上。
- VASP 与相关赝势库（`PMGVASPPSP`），确保具备合法使用授权。
- Slurm 作业调度系统，并配置好 `sbatch`、`squeue` 等命令。

### 2.2 Python 依赖

`pip install .` 会自动安装：

- CrewAI 与 FastMCP 生态；
- PyMatGen、ASE 等材料科学库；
- Quart/Flask、SQLAlchemy 等服务化依赖。

如需 GPU 或特殊版本的 LLM/Embedding 服务，需自行准备对应推理后端。

### 2.3 账号与密钥

- **Materials Project API Key**：用于结构搜索工具。
- **LLM API Key**：CrewAI 各 Agent 所调用的模型服务（OpenAI 兼容接口）。
- **Embedding API Key**：用于记忆与 RAG 检索（如 BAAI/bge-m3）。

---

## 3. 安装与目录布局

1. 克隆仓库并安装：

   ```bash
   git clone https://github.com/JiaxuanLiu-Arsko/VASPilot.git
   cd VASPilot
   pip install .
   ```

2. 准备工作目录（建议复制示例）：

   ```bash
   cp -r examples/1.Basic ~/vaspilot_example
   ```

   目录结构说明：

   - `configs/`：MCP 与 CrewAI 配置文件。
   - `mcp/attachment/`：包含 `slurm.sh`、`vdw_kernel.bindat` 等模板。
   - `mcp/work/`：MCP 运行时创建的计算目录。
   - `mcp/record/`：SQLite 数据库文件位置。
   - `crew_server/work/`：Web 服务缓存、日志与记忆库。

3. 设置环境变量（必须提前配置 VASP 赝势）：

   ```bash
   export PMG_VASP_PSP_DIR=/path/to/your/POTCARS
   ```

---

## 4. 配置说明

### 4.1 MCP (`configs/mcp_config.yaml`)

以 `examples/1.Basic/configs/mcp_config.yaml` 为模板，填写以下字段：

| 字段 | 说明 |
| ---- | ---- |
| `attachment_path` | 包含 Slurm 脚本与辅助文件的目录。 |
| `work_dir` | VASP 计算工作区；每次计算会在此创建独立子目录。 |
| `db_path` | SQLite 数据库文件路径，用于记录计算状态与解析结果。 |
| `mp_api_key` | Materials Project API Key。 |
| `structure_path` | 下载结构文件的存储位置。 |
| `VASP_default_INCAR` | 预设的不同计算类型 INCAR 参数，可按需修改。 |
| `enabled_tools` *(可选)* | 若只想启用部分 MCP 工具，可提供白名单列表。 |

### 4.2 CrewAI (`configs/crew_config.yaml`)

关键字段示例参考 `examples/1.Basic/configs/crew_config.yaml`：

- `llm_mapper`：定义可用 LLM 的访问方式（base_url、api_key、model、temperature）。
- `llm_config`：为每个 Agent 指定 `llm_mapper` 中的模型。
- `embbeder`：RAG 记忆所需的向量化接口。
- `mcp_server`：Crew 与 MCP 通信的地址与 `transport`。
- `agents`：
  - 配置每个 Agent 的 `goal`、`backstory` 与可用工具列表。
  - `manager_agent` 决定任务调度逻辑，其描述中应提及新增 Agent 的职责。
- `tool_params` *(可选)*：指定 `json_approx_search_tool`、`ask_question_tool` 等持久化工具的额外参数。

### 4.3 启动脚本（可选）

示例目录下提供了 `start_mcp_server.sh` 与 `start_crew_server.sh`。编辑其中的路径后，可在生产环境直接执行脚本完成启动。

---

## 5. 启动服务

### 5.1 启动 MCP 服务器

```bash
vaspilot_mcp \
  --config /path/to/configs/mcp_config.yaml \
  --port 8933 \
  --work-dir /path/to/mcp/workdir
```

成功启动后日志会显示 `🚀 启动VASP MCP服务器...`。若在远程机器运行，注意防火墙放通端口。

### 5.2 启动 CrewAI Web 服务（Quart）

```bash
vaspilot_quart \
  --config /path/to/configs/crew_config.yaml \
  --port 51293 \
  --work-dir /path/to/crew/workdir \
  --allow-path /path/to/vasp/projects \
  --max-concurrent-tasks 2 \
  --max-queue-size 10
```

参数说明：

- `--allow-path`：Web 前端允许用户上传文件的白名单目录。
- `--max-concurrent-tasks`：同一时间最多并行的任务数。
- `--max-queue-size`：等待队列长度，防止任务过载。

### 5.3 服务健康检查

1. 访问 `http://<host>:51293`，确认前端加载正常。
2. 在 UI 中创建简单任务（如结构检索），观察日志输出确认 Agent 与 MCP 通信正常。
3. 若使用 Slurm，确保任务提交目录含有 `slurm.sh`，并在计算节点正确执行。

---

## 6. 使用流程示例

1. **上传或检索结构**：在 Web 界面选择上传 POSCAR/CIF，或让系统调用 Materials Project。
2. **配置计算意图**：通过自然语言描述任务，例如“计算 2H-MoS2 的能带，弛豫阶段使用 IVDW=11”。
3. **任务执行**：经理 Agent 会调度晶体结构 Agent 与 VASP Agent；`wait_calc_tool` 会追踪 Slurm 作业直至完成。
4. **结果验证与汇报**：Result Validation Agent 读取数据库记录，确认所有计算成功并生成报告链接。
5. **下载结果**：在 Web 页面查看 log、结构文件与绘图；必要时可在 `mcp/work/` 下直接查看原始 VASP 文件。

> ⚠️ 建议在第一次部署时，先执行 `examples/1.Basic` 中的测试用例，确认目录权限、Slurm 队列与 POTCAR 配置无误。

---

## 7. 构建新的 Agent

大多数扩展场景只需修改 `crew_config.yaml` 即可完成新的智能体配置。具体步骤：

1. **准备模型映射**  
   在 `llm_mapper` 中添加新的模型条目。例如：

   ```yaml
   llm_mapper:
     reasoning-large:
       base_url: http://llm.internal:8000/v1
       api_key: sk-xxxx
       model: openai/Reasoner-Pro
       temperature: 0
   ```

2. **在 `llm_config` 中声明 Agent 与模型的绑定**：

   ```yaml
   llm_config:
     new_analysis_agent: reasoning-large
   ```

3. **新增 Agent 描述**：在 `agents` 段落加入自定义块。

   ```yaml
   agents:
     new_analysis_agent:
       role: Advanced Analyzer
       goal: "负责对 VASP 输出进行高级数据挖掘，例如态密度积分或自定义指标。"
       backstory: >
         你是材料信息学专家，熟悉 VASP 输出文件与数据分析流程。
       tools:
         - read_calc_results_from_db
         - python_plot
         - custom_feature_tool
   ```

4. **更新 `manager_agent` 的提示词**：确保在管理者提示词中说明新 Agent 的职责，避免调度缺失。

5. **在 Web 前端启用**：默认情况下，所有在配置文件中定义的 Agent 会被 `VaspCrew` 自动加载，无需修改源码。若需要在启动时注入上下文 ID 或持久化工具，可参考 `src/vaspilot/crew/vasp_crew.py` 中 `_inject_agent_tools` 的逻辑。

6. **测试**：重启 Crew 服务后，在前端通过任务描述或系统提示触发新 Agent 的行为，确认工具能够成功调用。

---

## 8. 添加新的工具

VASPilot 支持两类扩展工具：MCP 侧（通常用于涉及文件系统或长时计算的任务）与 Crew 本地工具（轻量逻辑、消息交互等）。

### 8.1 扩展 MCP 工具

1. **编写工具逻辑**：在 `src/vaspilot/tools/mcp/` 内创建新模块或在现有模块中添加函数。建议使用 `FastMCP` 的装饰器规范。

   ```python
   # src/vaspilot/tools/mcp/custom_tools.py
   from fastmcp import Context

   async def postprocess_bands(context: Context, vasprun_path: str) -> dict:
       # 解析 vasprun.xml 并返回所需指标
       ...
       return {"band_gap": band_gap, "metadata": ...}
   ```

2. **注册到 MCP 服务器**：在 `VASPMCPServer._register_tools()`（`src/vaspilot/tools/mcp/mcp_server.py`）中按示例添加：

   ```python
   if self._is_enabled("postprocess_bands"):
       @mcp.tool(name="postprocess_bands")
       async def postprocess_bands(vasprun_path: str) -> dict:
           return await custom_tools.postprocess_bands(Context(), vasprun_path)
   ```

3. **配置启用**：在 `mcp_config.yaml` 中添加到 `enabled_tools` 或保持默认（启用全部）。启动 MCP 服务后即可通过 Crew 调用。

4. **与 Agent 对接**：在 `crew_config.yaml` 的相关 Agent `tools` 列表中增加 `postprocess_bands`。

5. **测试**：通过 Web 前端或 `fastmcp` 客户端直接调用，检查返回结构是否符合预期，并确认数据库记录流程正常。

> ✅ 复用数据库：若需要写入计算状态，可调用 `VASPMCPServer.write_record` 保存结果，供 Result Validation Agent 读取。

### 8.2 添加 Crew 本地工具

本地工具适合轻量交互或非阻塞逻辑，例如向用户提问、轮询外部状态。

1. **创建工具类**：在 `src/vaspilot/tools/` 下新建文件，继承 `crewai.tools.BaseTool`，参考 `ask_user_tool.py` 或 `wait_calc_tool.py`。

   ```python
   # src/vaspilot/tools/custom_metric_tool.py
   from crewai.tools import BaseTool
   from pydantic import BaseModel, Field

   class CustomMetricInput(BaseModel):
       calc_id: str = Field(..., description="目标计算的 ID")

   class CustomMetricTool(BaseTool):
       args_schema = CustomMetricInput

       def __init__(self):
           super().__init__(
               name="custom_metric",
               description="根据计算 ID 读取数据库并返回自定义指标"
           )

       def _run(self, calc_id: str) -> dict:
           ...
   ```

2. **在 Crew 构建流程中注册**：修改 `VaspCrew._create_tools()`（`src/vaspilot/crew/vasp_crew.py`）将工具加入 `tool_dict`，或在初始化阶段通过配置驱动的方式注入。

   ```python
   from ..tools.custom_metric_tool import CustomMetricTool
   ...
   tool_dict["custom_metric"] = CustomMetricTool()
   ```

3. **更新 Agent 配置**：在 `crew_config.yaml` 中，将工具名称添加到目标 Agent 的 `tools` 列表。

4. **重启 Crew 服务并测试**：确保工具返回结构与 Agent 期望一致，必要时在日志中输出调试信息。

---

## 9. 常见问题排查

- **Slurm 提交失败**：检查 `mcp/attachment/` 是否包含正确的 `slurm.sh`，以及 `sbatch` 是否在 PATH 中。
- **POTCAR 找不到**：确认已设置 `PMG_VASP_PSP_DIR`，且 `potcar_map` 中使用的别名与目录一致。
- **Agent 不调用新工具**：确认工具名称在 MCP 注册、Crew `tool_dict` 中存在，并且写入 `agents.<agent_name>.tools` 列表。
- **RAG 相关报错**：检查 `crew_server/work/memory/` 权限，以及 Embedding 服务地址可达。
- **长时间无响应**：`wait_calc_tool` 默认轮询间隔 30 秒，可在工具实现中按需调整；同时检查 Slurm 队列是否拥塞。

---

## 10. 推荐操作流程

1. 在测试环境中使用 `examples/2.Basic+RAG` 验证记忆功能和新工具。
2. 将稳定配置固化为版本控制的 YAML，配合 `start_*.sh` 或 systemd 管理后台服务。
3. 通过数据库 (`mcp/record/record.db`) 定期备份计算元数据，避免任务信息丢失。
4. 扩展 Agent/工具后，编写最小化用例（结构、命令与期望输出），便于回归测试。

---

## 11. 附录：关键命令速查

| 目的 | 命令 |
| ---- | ---- |
| 启动 MCP 服务器 | `vaspilot_mcp --config configs/mcp_config.yaml --port 8933` |
| 启动 Quart Web 服务 | `vaspilot_quart --config configs/crew_config.yaml --port 51293` |
| 监听日志（示例） | `tail -f crew_server/work/output.log` |
| 清理旧计算目录 | `find mcp/work -maxdepth 1 -type d -mtime +7 -exec rm -rf {} +` *(执行前确认！)* |

---

通过以上步骤，您即可在集群环境中稳定运行 VASPilot，并根据科研或生产需求灵活扩展 Agent 能力与工具链。祝计算顺利！
