# XML表单驱动的DataFlow Agent系统示例

## 完整示例集合

### 示例1：医疗问答数据治理

```xml
<?xml version="1.0" encoding="UTF-8"?>
<workflow xmlns="http://dataflow.agent/workflow" 
          id="medical_qa_001" 
          version="1.0" 
          priority="high"
          created_at="2024-01-15T10:30:00Z">
    
    <topic>医疗问答数据</topic>
    <scene>医疗AI训练数据集处理</scene>
    
    <global_variables>
        <dataset_id>/data/medical_qa.json</dataset_id>
        <feedback>上次处理后发现数据重复率较高，需要加强去重；部分问答质量不高，需要质量评估</feedback>
        <history>
            2024-01-10: 完成基础数据格式转换
            2024-01-12: 进行了初步清洗，移除明显错误数据
            2024-01-14: 发现重复数据问题，需要进一步处理
        </history>
        
        <environment>
            <execution_mode>local</execution_mode>
            <resource_limits>
                <memory_limit>8GB</memory_limit>
                <cpu_limit>4</cpu_limit>
                <timeout>PT2H</timeout>
                <max_workers>4</max_workers>
            </resource_limits>
            <dependencies>
                <python_packages>
                    <package version=">=1.5.0">pandas</package>
                    <package version=">=0.25.0">scikit-learn</package>
                    <package version=">=3.4.1">nltk</package>
                </python_packages>
                <external_services>
                    <service>
                        <name>quality_assessment_api</name>
                        <endpoint>http://localhost:8888/quality</endpoint>
                        <auth_required>false</auth_required>
                    </service>
                </external_services>
            </dependencies>
        </environment>
        
        <custom_variables>
            <variable name="quality_threshold" type="float">0.85</variable>
            <variable name="similarity_threshold" type="float">0.9</variable>
            <variable name="min_length" type="integer">10</variable>
            <variable name="max_length" type="integer">500</variable>
            <variable name="language" type="string">zh</variable>
        </custom_variables>
    </global_variables>
    
    <pipeline_goal>
        对医疗问答数据进行深度清洗、去重、质量评估和增强，确保数据质量满足AI训练要求。
        具体目标：
        1. 去除重复和近似重复的问答对
        2. 评估问答质量，过滤低质量数据
        3. 规范化文本格式和医学术语
        4. 生成质量评分和置信度
        5. 输出符合训练标准的高质量数据集
    </pipeline_goal>
    
    <data_format>json</data_format>
    
    <output_requirements>
        输出高质量、无重复的医疗问答对，每条数据包含：
        - 原始问题和答案
        - 质量评分（0-1分）
        - 置信度评估
        - 处理状态标记
        - 医学领域分类标签
        要求：重复率&lt;2%，平均质量分&gt;0.85，数据完整率&gt;95%
    </output_requirements>
    
    <advanced_config>
        <pipeline_config>
            <parallel_execution>true</parallel_execution>
            <retry_on_failure>true</retry_on_failure>
            <max_retry_attempts>3</max_retry_attempts>
            <checkpoint_enabled>true</checkpoint_enabled>
        </pipeline_config>
        
        <quality_control>
            <validation_rules>
                <rule>
                    <name>completeness_check</name>
                    <description>检查问答对完整性</description>
                    <condition>question IS NOT NULL AND answer IS NOT NULL AND len(question) &gt; 5 AND len(answer) &gt; 10</condition>
                    <severity>error</severity>
                </rule>
                <rule>
                    <name>length_validation</name>
                    <description>检查文本长度合理性</description>
                    <condition>len(question) &lt; 200 AND len(answer) &lt; 1000</condition>
                    <severity>warning</severity>
                </rule>
                <rule>
                    <name>medical_relevance</name>
                    <description>检查医学相关性</description>
                    <condition>contains_medical_terms(question) OR contains_medical_terms(answer)</condition>
                    <severity>warning</severity>
                </rule>
            </validation_rules>
            
            <quality_metrics>
                <completeness_threshold>0.95</completeness_threshold>
                <accuracy_threshold>0.90</accuracy_threshold>
                <consistency_threshold>0.85</consistency_threshold>
                <duplication_threshold>0.02</duplication_threshold>
            </quality_metrics>
            
            <error_handling>
                <continue_on_error>true</continue_on_error>
                <error_log_level>warning</error_log_level>
                <notification_enabled>true</notification_enabled>
            </error_handling>
        </quality_control>
        
        <monitoring>
            <metrics_collection>true</metrics_collection>
            <performance_tracking>true</performance_tracking>
            <resource_monitoring>true</resource_monitoring>
            <alert_thresholds>
                <execution_time_threshold>PT90M</execution_time_threshold>
                <memory_usage_threshold>0.80</memory_usage_threshold>
                <error_rate_threshold>0.05</error_rate_threshold>
            </alert_thresholds>
        </monitoring>
        
        <debug_config>
            <debug_mode>false</debug_mode>
            <verbose_logging>true</verbose_logging>
            <step_by_step_execution>false</step_by_step_execution>
            <intermediate_results_saving>true</intermediate_results_saving>
        </debug_config>
    </advanced_config>
</workflow>
```

### 示例2：财务数据异常检测

```xml
<?xml version="1.0" encoding="UTF-8"?>
<workflow xmlns="http://dataflow.agent/workflow" 
          id="finance_anomaly_001" 
          version="1.0" 
          priority="urgent">
    
    <topic>财务报表数据</topic>
    <scene>财务异常检测和风险评估</scene>
    
    <global_variables>
        <dataset_id>/data/financial_reports_2024.xlsx</dataset_id>
        <feedback>上季度发现多项异常指标，需要加强异常检测算法的敏感度</feedback>
        <history>
            已建立基础财务指标监控体系
            历史异常检测准确率85%
            需要提升对新型风险模式的识别能力
        </history>
        
        <environment>
            <execution_mode>distributed</execution_mode>
            <resource_limits>
                <memory_limit>16GB</memory_limit>
                <timeout>PT4H</timeout>
                <max_workers>8</max_workers>
            </resource_limits>
        </environment>
        
        <custom_variables>
            <variable name="anomaly_threshold" type="float">0.95</variable>
            <variable name="risk_level" type="string">medium</variable>
            <variable name="lookback_periods" type="integer">12</variable>
            <variable name="confidence_interval" type="float">0.99</variable>
        </custom_variables>
    </global_variables>
    
    <pipeline_goal>
        建立全面的财务异常检测体系，识别潜在风险和异常模式：
        1. 多维度异常检测（时间序列、横截面、业务逻辑）
        2. 风险评估和预警机制
        3. 异常原因分析和解释
        4. 生成详细的风险评估报告
    </pipeline_goal>
    
    <data_format>xlsx</data_format>
    
    <output_requirements>
        生成完整的异常检测报告，包含：
        - 异常指标识别和风险评级
        - 时间序列异常点标记
        - 业务逻辑一致性检查结果
        - 风险预警建议
        要求：检测准确率&gt;90%，误报率&lt;5%
    </output_requirements>
    
    <advanced_config>
        <quality_control>
            <validation_rules>
                <rule>
                    <name>financial_ratio_check</name>
                    <description>检查财务比率合理性</description>
                    <condition>debt_ratio &lt; 1.0 AND current_ratio &gt; 0.5</condition>
                    <severity>error</severity>
                </rule>
                <rule>
                    <name>temporal_consistency</name>
                    <description>检查时间序列一致性</description>
                    <condition>abs(current_value - moving_average) &lt; 3 * standard_deviation</condition>
                    <severity>warning</severity>
                </rule>
            </validation_rules>
            
            <quality_metrics>
                <accuracy_threshold>0.90</accuracy_threshold>
                <completeness_threshold>0.98</completeness_threshold>
            </quality_metrics>
        </quality_control>
        
        <monitoring>
            <metrics_collection>true</metrics_collection>
            <alert_thresholds>
                <execution_time_threshold>PT3H</execution_time_threshold>
                <error_rate_threshold>0.02</error_rate_threshold>
            </alert_thresholds>
        </monitoring>
    </advanced_config>
</workflow>
```

### 示例3：电商用户行为分析

```xml
<?xml version="1.0" encoding="UTF-8"?>
<workflow xmlns="http://dataflow.agent/workflow" 
          id="ecommerce_behavior_001" 
          version="1.0" 
          priority="normal">
    
    <topic>电商用户行为数据</topic>
    <scene>用户画像构建和个性化推荐数据准备</scene>
    
    <global_variables>
        <dataset_id>/data/user_behavior_logs.jsonl</dataset_id>
        <feedback>需要更精准的用户分群，提升推荐系统效果</feedback>
        <history>已完成基础数据清洗和用户ID统一</history>
        
        <environment>
            <execution_mode>hybrid</execution_mode>
            <resource_limits>
                <memory_limit>32GB</memory_limit>
                <timeout>PT6H</timeout>
            </resource_limits>
        </environment>
        
        <custom_variables>
            <variable name="min_session_duration" type="integer">30</variable>
            <variable name="max_session_gap" type="integer">1800</variable>
            <variable name="feature_time_window" type="integer">30</variable>
            <variable name="user_active_threshold" type="integer">5</variable>
        </custom_variables>
    </global_variables>
    
    <pipeline_goal>
        构建全面的用户行为分析体系：
        1. 用户会话识别和行为序列构建
        2. 多维度特征工程（时间、偏好、频次等）
        3. 用户分群和画像标签生成
        4. 个性化推荐候选集准备
        5. 用户价值评估和生命周期分析
    </pipeline_goal>
    
    <data_format>jsonl</data_format>
    
    <output_requirements>
        输出结构化的用户画像数据，包含：
        - 用户基础属性和行为特征
        - 兴趣偏好和品类倾向
        - 购买力和生命周期阶段
        - 个性化推荐特征向量
        要求：特征覆盖率&gt;90%，用户分群准确率&gt;85%
    </output_requirements>
    
    <advanced_config>
        <pipeline_config>
            <parallel_execution>true</parallel_execution>
            <checkpoint_enabled>true</checkpoint_enabled>
        </pipeline_config>
        
        <quality_control>
            <validation_rules>
                <rule>
                    <name>session_validity</name>
                    <description>检查用户会话有效性</description>
                    <condition>session_duration &gt; 30 AND page_views &gt; 1</condition>
                    <severity>warning</severity>
                </rule>
                <rule>
                    <name>behavior_consistency</name>
                    <description>检查行为序列一致性</description>
                    <condition>timestamp_sequence_is_valid(behaviors)</condition>
                    <severity>error</severity>
                </rule>
            </validation_rules>
            
            <quality_metrics>
                <completeness_threshold>0.90</completeness_threshold>
                <consistency_threshold>0.95</consistency_threshold>
            </quality_metrics>
        </quality_control>
        
        <monitoring>
            <performance_tracking>true</performance_tracking>
            <resource_monitoring>true</resource_monitoring>
        </monitoring>
    </advanced_config>
</workflow>
```

### 示例4：文本情感分析数据预处理

```xml
<?xml version="1.0" encoding="UTF-8"?>
<workflow xmlns="http://dataflow.agent/workflow" 
          id="sentiment_analysis_001" 
          version="1.0" 
          priority="normal">
    
    <topic>社交媒体文本情感数据</topic>
    <scene>情感分析模型训练数据预处理</scene>
    
    <global_variables>
        <dataset_id>/data/social_media_posts.csv</dataset_id>
        <feedback>标注质量参差不齐，需要标注一致性检查</feedback>
        <history>已收集10万条社交媒体文本，完成初步人工标注</history>
        
        <custom_variables>
            <variable name="min_text_length" type="integer">5</variable>
            <variable name="max_text_length" type="integer">280</variable>
            <variable name="annotation_confidence" type="float">0.8</variable>
            <variable name="language_filter" type="list">["zh", "en"]</variable>
        </custom_variables>
    </global_variables>
    
    <pipeline_goal>
        构建高质量的情感分析训练数据集：
        1. 文本标准化和清洗（去噪、规范化）
        2. 情感标注质量检查和一致性验证
        3. 数据增强和类别平衡处理
        4. 特征提取和向量化准备
        5. 训练集/验证集/测试集划分
    </pipeline_goal>
    
    <data_format>csv</data_format>
    
    <output_requirements>
        输出平衡的情感分析训练数据，包含：
        - 清洗后的文本内容
        - 高置信度的情感标签（正面/负面/中性）
        - 文本特征向量
        - 数据集划分标记
        要求：标注一致性&gt;90%，类别平衡度&gt;80%
    </output_requirements>
</workflow>
```

### 示例5：时间序列预测数据准备

```xml
<?xml version="1.0" encoding="UTF-8"?>
<workflow xmlns="http://dataflow.agent/workflow" 
          id="timeseries_forecast_001" 
          version="1.0" 
          priority="high">
    
    <topic>销售时间序列数据</topic>
    <scene>销售预测模型数据准备</scene>
    
    <global_variables>
        <dataset_id>/data/daily_sales_2019_2024.parquet</dataset_id>
        <feedback>存在季节性和趋势性，需要特征工程支持</feedback>
        <history>历史数据显示明显的周期性模式和异常值</history>
        
        <environment>
            <execution_mode>local</execution_mode>
            <resource_limits>
                <memory_limit>16GB</memory_limit>
                <timeout>PT3H</timeout>
            </resource_limits>
        </environment>
        
        <custom_variables>
            <variable name="forecast_horizon" type="integer">30</variable>
            <variable name="lookback_window" type="integer">365</variable>
            <variable name="seasonality_period" type="list">[7, 30, 365]</variable>
            <variable name="outlier_threshold" type="float">3.0</variable>
        </custom_variables>
    </global_variables>
    
    <pipeline_goal>
        准备高质量的时间序列预测数据：
        1. 时间序列数据质量检查和异常值处理
        2. 趋势分解和季节性分析
        3. 特征工程（滞后特征、滑动统计、节假日等）
        4. 数据标准化和归一化
        5. 训练窗口和预测窗口构建
    </pipeline_goal>
    
    <data_format>parquet</data_format>
    
    <output_requirements>
        输出结构化的时间序列建模数据，包含：
        - 清洗后的时间序列数据
        - 丰富的时间特征和外部特征
        - 训练样本和标签对
        - 数据统计信息和质量报告
        要求：数据连续性&gt;95%，特征完整性&gt;90%
    </output_requirements>
    
    <advanced_config>
        <quality_control>
            <validation_rules>
                <rule>
                    <name>temporal_continuity</name>
                    <description>检查时间序列连续性</description>
                    <condition>missing_dates_ratio &lt; 0.05</condition>
                    <severity>error</severity>
                </rule>
                <rule>
                    <name>outlier_detection</name>
                    <description>检测异常值</description>
                    <condition>abs(value - rolling_mean) &lt; 3 * rolling_std</condition>
                    <severity>warning</severity>
                </rule>
            </validation_rules>
        </quality_control>
    </advanced_config>
</workflow>
```

## 使用这些示例

### 1. 直接使用示例

将示例XML复制到系统的XML表单输入框中，点击"执行工作流"即可运行。

### 2. 自定义修改

根据您的具体需求修改示例中的：
- `dataset_id`：更改为您的数据集路径
- `custom_variables`：调整参数值
- `pipeline_goal`：修改治理目标
- `validation_rules`：添加或修改验证规则

### 3. 生成新示例

通过Former Agent对话，描述您的具体需求，系统会自动生成适合的XML表单。

### 4. 示例场景说明

| 示例 | 适用场景 | 主要功能 | 输出结果 |
|------|----------|----------|----------|
| 医疗问答 | AI训练数据 | 去重、质量评估 | 高质量问答对 |
| 财务异常 | 风险监控 | 异常检测、预警 | 风险评估报告 |
| 用户行为 | 个性化推荐 | 特征工程、分群 | 用户画像数据 |
| 情感分析 | NLP模型训练 | 文本清洗、标注验证 | 平衡训练集 |
| 时间序列 | 预测建模 | 特征工程、异常处理 | 建模数据集 |

这些示例展示了XML表单驱动系统的强大能力和灵活性，可以适配各种不同的数据治理场景。
