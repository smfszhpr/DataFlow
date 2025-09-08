# XML表单驱动的DataFlow Agent系统

## 概述

本系统实现了基于XML表单的DataFlow Agent架构，包含两个核心组件：

1. **Former Agent（需求收集代理）**：通过多轮对话引导用户明确数据治理需求，自动生成结构化的XML表单
2. **Worker Agent（执行代理）**：基于XML表单自动构建和执行DataFlow数据治理pipeline

## 架构设计

```
用户输入 → Former Agent → XML表单 → Worker Agent → DataFlow Pipeline → 执行结果
   ↑         ↓                                           ↓
   ←── 多轮对话交互 ←─────────────────────────────── 反馈和结果展示
```

### 核心组件

#### 1. Former Agent（需求收集代理）
- **功能**：通过对话收集用户需求
- **职责**：
  - 引导用户明确数据集信息（主题、场景、格式等）
  - 收集数据治理目标和质量要求
  - 获取历史处理记录和用户反馈
  - 自动生成标准化XML表单

#### 2. Worker Agent（执行代理）
- **功能**：执行基于XML表单的工作流
- **职责**：
  - 解析XML表单内容
  - 构建ChatAgentRequest对象
  - 调用DataFlow任务链执行pipeline
  - 返回执行结果和生成的代码

## XML表单规范

### 基础结构

```xml
<workflow id="unique_id" version="1.0" priority="normal">
    <!-- 数据集主题内容 -->
    <topic>医疗问答数据</topic>
    
    <!-- 使用场景描述 -->
    <scene>医疗AI训练数据集处理</scene>
    
    <!-- 全局变量配置 -->
    <global_variables>
        <dataset_id>/data/medical_qa.json</dataset_id>
        <feedback>上次处理后发现数据重复率较高，需要加强去重</feedback>
        <history>已进行过基础清洗和格式化</history>
        
        <!-- 可选的环境配置 -->
        <environment>
            <execution_mode>local</execution_mode>
            <resource_limits>
                <memory_limit>8GB</memory_limit>
                <timeout>PT2H</timeout>
            </resource_limits>
        </environment>
        
        <!-- 自定义变量 -->
        <custom_variables>
            <variable name="quality_threshold" type="float">0.85</variable>
            <variable name="batch_size" type="integer">1000</variable>
        </custom_variables>
    </global_variables>
    
    <!-- Pipeline治理目标 -->
    <pipeline_goal>对医疗问答数据进行深度清洗、去重、质量评估和增强</pipeline_goal>
    
    <!-- 数据格式 -->
    <data_format>json</data_format>
    
    <!-- 输出要求 -->
    <output_requirements>输出高质量、无重复的医疗问答对，包含质量评分</output_requirements>
    
    <!-- 高级配置（可选） -->
    <advanced_config>
        <pipeline_config>
            <parallel_execution>true</parallel_execution>
            <retry_on_failure>true</retry_on_failure>
            <max_retry_attempts>3</max_retry_attempts>
        </pipeline_config>
        
        <quality_control>
            <validation_rules>
                <rule>
                    <name>completeness_check</name>
                    <description>检查数据完整性</description>
                    <condition>missing_ratio &lt; 0.05</condition>
                    <severity>error</severity>
                </rule>
            </validation_rules>
            
            <quality_metrics>
                <completeness_threshold>0.95</completeness_threshold>
                <accuracy_threshold>0.90</accuracy_threshold>
                <duplication_threshold>0.02</duplication_threshold>
            </quality_metrics>
        </quality_control>
        
        <monitoring>
            <metrics_collection>true</metrics_collection>
            <performance_tracking>true</performance_tracking>
            <alert_thresholds>
                <execution_time_threshold>PT30M</execution_time_threshold>
                <memory_usage_threshold>0.80</memory_usage_threshold>
            </alert_thresholds>
        </monitoring>
        
        <debug_config>
            <debug_mode>false</debug_mode>
            <verbose_logging>true</verbose_logging>
            <intermediate_results_saving>true</intermediate_results_saving>
        </debug_config>
    </advanced_config>
</workflow>
```

### 支持的数据格式

- `json` - JSON格式
- `csv` - 逗号分隔值
- `parquet` - 列式存储格式
- `xlsx` - Excel文件
- `xml` - XML格式
- `txt` - 纯文本
- `jsonl` - JSON Lines格式
- `avro` - Apache Avro格式
- `hdf5` - HDF5格式
- `pickle` - Python Pickle格式

### 执行模式

- `local` - 本地执行
- `distributed` - 分布式执行
- `cloud` - 云端执行
- `hybrid` - 混合模式

## 使用指南

### 1. 环境准备

确保已安装DataFlow及其依赖：

```bash
cd /Users/zyd/DataFlow
pip install -e .
```

### 2. 启动系统

```bash
cd /Users/zyd/DataFlow/test
python xml_form_driven_agent.py
```

系统将在以下地址启动：
- Web界面：http://localhost:8080
- API文档：http://localhost:8080/api/docs

### 3. 配置API

在界面中配置：
- **API Key**：您的LLM服务API密钥
- **API URL**：Chat API服务地址
- **模型**：选择合适的语言模型

### 4. 对话收集需求

通过Former Agent进行多轮对话：

**示例对话流程：**

```
用户：我想处理一批医疗问答数据，用于训练AI模型
Agent：好的！请告诉我您的数据集具体包含什么类型的医疗问答？是专科问答还是通用医疗咨询？

用户：主要是心血管疾病相关的问答，大概有10万条记录
Agent：了解了。您的数据集当前是什么格式？存储在哪里？

用户：JSON格式，在/data/cardiology_qa.json文件中
Agent：很好。您希望对这些数据进行什么样的治理？比如清洗、去重、质量评估等？

用户：之前处理过一次，发现重复数据比较多，还有一些问答质量不高
Agent：明白了。您希望达到什么样的质量标准？输出需要包含哪些信息？

用户：希望去除重复，保留高质量问答，最好能给每条数据评分
Agent：完美！基于您的需求，我已为您生成了完整的XML工作流表单...
```

### 5. 执行工作流

Former Agent生成XML表单后，点击"执行工作流"按钮，Worker Agent将：

1. 解析XML表单内容
2. 构建DataFlow任务链
3. 执行数据治理pipeline
4. 返回执行结果和生成的代码

## API接口

### Former Agent API

```http
POST /api/former-agent
Content-Type: application/json

{
    "user_query": "我想处理医疗问答数据",
    "conversation_history": [],
    "session_id": "optional_session_id",
    "api_key": "your_api_key",
    "chat_api_url": "your_api_url",
    "model": "deepseek-v3"
}
```

**响应：**
```json
{
    "need_more_info": true,
    "agent_response": "好的！请告诉我您的数据集具体信息...",
    "xml_form": null,
    "conversation_history": [...]
}
```

### Worker Agent API

```http
POST /api/worker-agent
Content-Type: application/json

{
    "xml_content": "<workflow>...</workflow>",
    "api_key": "your_api_key",
    "chat_api_url": "your_api_url",
    "model": "deepseek-v3"
}
```

## 示例场景

### 场景1：文本分类数据清洗

```xml
<workflow>
    <topic>新闻文本分类</topic>
    <scene>新闻分类模型训练数据预处理</scene>
    <global_variables>
        <dataset_id>/data/news_classification.csv</dataset_id>
        <feedback>标签分布不均衡，需要数据平衡处理</feedback>
        <history>已完成基础格式转换</history>
    </global_variables>
    <pipeline_goal>清洗新闻文本，平衡类别分布，提取特征</pipeline_goal>
    <data_format>csv</data_format>
    <output_requirements>平衡的训练集，包含文本特征和标签</output_requirements>
</workflow>
```

### 场景2：财务数据质量评估

```xml
<workflow>
    <topic>财务报表数据</topic>
    <scene>财务分析系统数据质量控制</scene>
    <global_variables>
        <dataset_id>/data/financial_reports.xlsx</dataset_id>
        <feedback>发现部分数据异常值，需要异常检测</feedback>
        <history>已进行基础数据验证</history>
    </global_variables>
    <pipeline_goal>检测异常值，验证数据一致性，生成质量报告</pipeline_goal>
    <data_format>xlsx</data_format>
    <output_requirements>清洁的财务数据和详细的质量评估报告</output_requirements>
</workflow>
```

### 场景3：图像标注数据增强

```xml
<workflow>
    <topic>图像分类标注</topic>
    <scene>计算机视觉模型训练数据准备</scene>
    <global_variables>
        <dataset_id>/data/image_labels.json</dataset_id>
        <feedback>数据集较小，需要数据增强</feedback>
        <history>已完成基础标注验证</history>
    </global_variables>
    <pipeline_goal>验证标注质量，生成数据增强策略，扩充数据集</pipeline_goal>
    <data_format>json</data_format>
    <output_requirements>高质量的扩充数据集，包含原始和增强数据</output_requirements>
</workflow>
```

## 扩展开发

### 添加自定义验证规则

```python
class CustomValidationRule:
    def __init__(self, name: str, condition: str, severity: str):
        self.name = name
        self.condition = condition
        self.severity = severity
    
    def validate(self, data: Any) -> bool:
        # 实现自定义验证逻辑
        pass
```

### 集成新的数据源

```python
class CustomDataSource:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def load_data(self, dataset_id: str) -> Any:
        # 实现自定义数据加载逻辑
        pass
```

### 添加新的执行模式

```python
class CustomExecutionMode:
    def __init__(self, mode_config: Dict[str, Any]):
        self.config = mode_config
    
    async def execute_pipeline(self, workflow: WorkflowXML) -> ChatResponse:
        # 实现自定义执行逻辑
        pass
```

## 故障排除

### 常见问题

1. **XML解析失败**
   - 检查XML格式是否正确
   - 验证必需字段是否完整
   - 确认字符编码为UTF-8

2. **Former Agent对话异常**
   - 检查API配置是否正确
   - 验证网络连接
   - 确认API密钥有效

3. **Worker Agent执行失败**
   - 检查数据集路径是否存在
   - 验证权限配置
   - 查看执行日志获取详细错误信息

### 日志查看

系统日志位于：
- Former Agent日志：控制台输出
- Worker Agent日志：`/tmp/dataflow_worker_agent.log`
- Pipeline执行日志：`generated_pipelines/` 目录下

### 性能优化

1. **内存优化**
   - 配置合适的`memory_limit`
   - 使用批处理模式处理大数据集
   - 启用中间结果缓存

2. **执行速度优化**
   - 启用并行执行
   - 配置合适的`max_workers`
   - 使用分布式执行模式

## 更新日志

### v1.0.0 (当前版本)
- 实现Former Agent对话收集功能
- 支持XML表单自动生成
- 集成Worker Agent执行引擎
- 提供Web界面和API接口
- 支持多种数据格式和执行模式

## 许可证

本项目遵循DataFlow项目的许可证协议。
