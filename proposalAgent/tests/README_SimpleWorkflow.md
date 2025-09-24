# 简化工作流使用指南

一个简化但完整的项目评估工作流，保持与setup相同的流程，能够正常运行并生成详细的分析报告。

## 🚀 快速开始

### 1. 基本用法

```python
from simple_proposal_runner import create_simple_runner

# 创建运行器
runner = create_simple_runner()

# 方法1：直接使用graph.invoke()
input_data = {
    "messages": [("user", "请对区块链技术在金融行业的应用进行项目评估分析")],
    "research_topic": ["区块链", "金融"],
    "filepath": ""
}
config = {"configurable": {"thread_id": "my_thread_123"}}
result = runner.graph.invoke(input_data, config=config)

# 方法2：使用便捷方法（推荐）
result = runner.evaluate(
    user_prompt="请评估人工智能在教育领域的应用项目",
    user_interests=["技术可行性", "教育效果", "实施成本"],
    filepath=""
)
```

### 2. 完整示例

```python
# 运行完整示例
python example_usage.py
```

## 📋 功能特性

### ✅ 已实现功能

- **完整的评估流程**：从意图识别到最终报告生成
- **多维度分析**：
  - 学术价值评估
  - 社会影响分析
  - 未来影响力预测
  - 跨学科协同分析
  - 可行性与创新性辩论
- **智能路由**：自动判断是直接回答还是进行深度分析
- **错误处理**：完善的异常处理和fallback机制
- **记忆系统**：支持持久化记忆和上下文管理

### 📊 输出报告包含

1. **技术可行性分析**（权重可调）
2. **市场前景分析**（权重可调）
3. **学术研究基础评估**
4. **未来影响力预测**
5. **社会影响分析**
6. **跨学科协同能力评估**
7. **综合评分与建议**

## 🔧 配置说明

### 模型配置

工作流支持多种LLM提供商：

```python
# 默认使用TONGYI_CONFIG（通义千问）
TONGYI_CONFIG = {
    "llm_provider": "tongyi",
    "api_key": "your-api-key",
    "deep_think_llm": "qwen-plus",
    "quick_think_llm": "qwen-plus",
    "backend_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

# 也支持OpenAI、Google等
```

### 自定义配置

```python
custom_config = {
    "llm_provider": "openai", 
    "api_key": "your-openai-key",
    "deep_think_llm": "gpt-4",
    "quick_think_llm": "gpt-3.5-turbo"
}

runner = create_simple_runner(config=custom_config)
```

## 📈 测试结果示例

### 成功案例

```
=== 测试便捷方法 ===
✓ 便捷方法调用成功
状态: completed
线程ID: simple_eval_-6394607110803844173_2025-09-23
✓ 评估完成

=== 最终评估报告已生成 ===
报告长度: 4716 字符
```

### 报告内容示例

```
区块链技术在金融行业的可行性评估：

一、技术可行性分析（权重 0.4）
优势：
1. 去中心化架构：减少对中心化金融机构的依赖...
2. 不可篡改性：交易记录一经写入即不可更改...
...

四、总体评分（加权综合）
| 维度               | 得分（满分1.0） | 权重 | 加权得分 |
|--------------------|------------------|------|----------|
| 技术可行性         | 0.6              | 0.4  | 0.24     |
| 市场前景           | 0.7              | 0.4  | 0.28     |
| 总分               |                  |      | 0.84     |
```

## 🛠️ 安装依赖

```bash
pip install chromadb langgraph langchain-openai langchain-community
```

## 📝 使用场景

### 适用于

- ✅ 项目可行性评估
- ✅ 学术论文评估  
- ✅ 技术方案分析
- ✅ 市场前景预测
- ✅ 跨学科项目评估

### 输入格式

#### 基本输入
```python
{
    "messages": [("user", "你的评估请求")],
    "research_topic": ["关键词1", "关键词2"],
    "filepath": "可选的PDF文件路径"
}
```

#### 高级选项
```python
runner.evaluate(
    user_prompt="评估请求",
    user_interests=["关注点1", "关注点2"],  # 可选
    filepath="path/to/document.pdf"        # 可选
)
```

## 🔍 工作流程

```
用户输入 → 意图识别 → 路由决策
    ↓
简单问答 ← → 深度分析流程
    ↓            ↓
直接输出    结构化分析 → 调度规划
                ↓
            并行信息收集（学术、社会、未来影响、跨学科）
                ↓
            辩论分析（可行性、创新性）
                ↓
            综合分析 → 完备性检查 → 最终报告生成
```

## 📚 API参考

### SimpleProposalRunner类

#### 方法

- `evaluate(user_prompt, user_interests=None, filepath="")` - 便捷评估方法
- `get_info()` - 获取工作流信息
- `graph.invoke(input_data, config)` - 直接调用图

#### 返回格式

```python
{
    "status": "completed",  # completed | error | interrupted
    "thread_id": "unique_thread_id",
    "final_report": "详细评估报告...",
    "analysis_summary": "分析摘要...",
    "academic_analysis": "学术分析...",
    "social_analysis": "社会影响分析...",
    "future_influence": "未来影响分析...",
    "debate_results": {"可行性": "...", "创新性": "..."},
    "full_result": {...}  # 完整状态
}
```

## ⚠️ 注意事项

1. **API密钥**：确保在`model_config.py`中配置了正确的API密钥
2. **网络连接**：需要稳定的网络连接访问LLM服务
3. **文件格式**：PDF文件需要是可读的格式
4. **内存使用**：大型模型可能消耗较多内存
5. **并发限制**：注意API调用频率限制

## 🔄 更新日志

### v1.0 - 当前版本
- ✅ 完整工作流实现
- ✅ 多LLM提供商支持
- ✅ 错误处理机制
- ✅ 便捷API接口
- ✅ 记忆系统集成
- ✅ 并行处理优化

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工作流！

## 📄 许可证

该项目遵循原项目的许可证条款。

---

**🎉 现在你就可以开始使用简化工作流进行项目评估了！**
