# VASPilot

基于CrewAI智能代理系统的VASP计算自动化工具，集成MCP（Model Control Protocol）协议，为材料科学研究提供智能化的计算工作流程。

## 项目特性

- 🤖 **智能代理系统**: 基于CrewAI框架的多智能体协作，自动化VASP计算流程
- 🔌 **MCP协议支持**: 符合Model Control Protocol标准，提供标准化的工具接口
- 🧮 **全面的VASP计算**: 支持结构优化、SCF、NSCF、能带计算等多种计算类型
- 🔍 **材料数据库集成**: 集成Materials Project API，便于材料搜索和结构获取
- 📊 **可视化分析**: 内置matplotlib绘图工具，支持能带图、态密度图等分析
- 🌐 **Web界面**: 提供Flask Web服务器，支持浏览器操作
- ⚡ **异步处理**: 支持长时间运行的计算任务管理
- 📝 **数据库管理**: SQLite数据库存储计算记录和结果

## 系统架构

```
用户请求 → CrewAI智能代理 → MCP工具服务 → VASP计算 → 结果分析
    ↓              ↓              ↓            ↓           ↓
Web界面        多代理协作      标准化接口    HPC集群     可视化展示
   ↓              ↓              ↓            ↓           ↓
Flask服务      任务规划        工具调用      任务提交     图表生成
```

## 核心组件

### 1. 智能代理系统
- **晶体结构代理**: 负责结构搜索、分析和操作
- **VASP计算代理**: 执行各种VASP计算任务
- **结果验证代理**: 验证和分析计算结果

### 2. MCP工具服务
- **VASP计算工具**: 结构优化、SCF、NSCF计算
- **材料搜索工具**: Materials Project数据库集成
- **结构分析工具**: 晶体结构分析和对称化
- **可视化工具**: Python绘图和数据分析

### 3. Web服务器
- **Flask界面**: 用户友好的Web操作界面
- **任务管理**: 计算任务的提交、监控和管理
- **结果展示**: 计算结果的可视化展示

## 安装和配置

### 1. 安装项目

```bash
# 克隆项目到本地
git clone https://github.com/your-username/vaspilot.git
cd vaspilot

# 安装项目（开发模式）
pip install -e .
```

### 2. 配置文件

项目提供了两个主要配置文件：

#### CrewAI配置 (`configs/crew_config.yaml`)
配置智能代理系统、LLM连接和嵌入模型：

```yaml
llm_mapper:
  deepseek-v3:
    base_url: http://your-llm-server:8000/v1
    api_key: your-api-key
    model: your-model-name
    temperature: 0

mcp_server:
  url: http://localhost:8933/mcp
  transport: streamable-http

agents:
  crystal_structure_agent:
    goal: "根据需要，利用工具搜索、分析或操作晶体结构"
    backstory: "经验丰富的晶体结构专家"
```

#### MCP配置 (`configs/mcp_config.yaml`)
配置VASP计算环境和参数：

```yaml
POTCAR_dir: /path/to/POTCARS/
VASP_cmd: srun -N 1 mpirun -np 64 vasp_ncl
work_dir: /path/to/work/directory
db_path: /path/to/database.db
mp_api_key: your-materials-project-api-key

VASP_default_INCAR:
  relaxation:
    PREC: 'Accurate'
    ISMEAR: 0
    SIGMA: 0.03
    EDIFF: 1e-6
    IBRION: 1
    EDIFFG: -0.005
    NSW: 100
```

## 使用方法

### 1. 启动MCP服务

```bash
# 启动MCP工具服务器（默认端口8933）
vaspilot_mcp

# 自定义配置和端口
vaspilot_mcp --config configs/mcp_config.yaml --port 8933 --host 0.0.0.0
```

### 2. 启动CrewAI服务

```bash
# 启动CrewAI Web服务器（默认端口51293）
vaspilot_server

# 自定义配置和端口
vaspilot_server --config configs/crew_config.yaml --port 51293 --host 0.0.0.0
```

### 3. Web界面使用

访问 `http://localhost:51293` 打开Web界面，可以：
- 提交VASP计算任务
- 监控计算进度
- 查看和下载结果
- 可视化分析数据

### 4. 支持的计算类型

#### 结构优化
```python
# 自动结构优化
"对TiO2进行结构优化计算"
```

#### 能带计算
```python
# SCF + NSCF能带计算
"计算MoS2的能带结构"
```

#### 材料搜索
```python
# Materials Project搜索
"搜索带隙在1-3eV之间的钙钛矿材料"
```

## 命令行参数

### vaspilot_mcp 参数
- `--config`: MCP配置文件路径（默认：configs/mcp_config.yaml）
- `--port`: 服务器端口（默认：8933）
- `--host`: 服务器地址（默认：0.0.0.0）
- `--work-dir`: 工作目录（默认：当前目录）
- `--debug`: 开启调试模式

### vaspilot_server 参数
- `--config`: CrewAI配置文件路径（默认：configs/crew_config.yaml）
- `--host`: 服务器地址（默认：0.0.0.0）
- `--port`: 服务器端口（默认：51293）
- `--work-dir`: 工作目录（默认：当前目录）
- `--allow-path`: 允许访问的目录路径
- `--debug`: 开启调试模式

## 开发和扩展

### 项目结构
```
src/vaspilot/
├── scripts/          # 启动脚本
├── server/           # Web服务器
│   └── flask_server/ # Flask实现
├── tools/            # 工具集合
│   └── mcp/         # MCP工具实现
├── crew/            # CrewAI代理定义
└── listener/        # 事件监听器
```

### 添加新工具

1. 在 `src/vaspilot/tools/mcp/` 下创建新的工具模块
2. 在MCP服务器中注册新工具
3. 更新配置文件中的工具列表

### 自定义代理

1. 在 `configs/crew_config.yaml` 中定义新代理
2. 配置代理的目标、背景和使用的LLM
3. 重启服务器应用更改

## 故障排除

### 常见问题

1. **MCP服务连接失败**
   - 检查MCP服务器是否已启动
   - 确认端口配置正确
   - 检查防火墙设置

2. **VASP计算失败**
   - 检查POTCAR路径配置
   - 确认VASP命令可执行
   - 检查HPC环境设置

3. **LLM连接问题**
   - 验证API密钥和端点URL
   - 检查网络连接
   - 确认模型名称正确

### 调试技巧

1. 使用 `--debug` 参数启动服务器
2. 查看日志文件了解详细错误信息
3. 检查数据库中的任务状态
4. 验证配置文件格式

## 依赖项

主要依赖包：
- `crewai>=0.102.0` - 智能代理框架
- `fastmcp>=2.10.6` - MCP协议实现
- `pymatgen>=2025.0.0` - 材料科学工具
- `ase>=3.22.0` - 原子结构环境
- `flask` - Web框架
- `numpy`, `pandas`, `matplotlib` - 数据分析和可视化

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 致谢

- [CrewAI](https://github.com/joaomdmoura/crewAI) - 智能代理框架
- [Pymatgen](https://pymatgen.org/) - 材料科学计算工具
- [Materials Project](https://materialsproject.org/) - 材料数据库 