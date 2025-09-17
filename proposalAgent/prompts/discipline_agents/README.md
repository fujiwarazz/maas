# 学科专业Agent系统

## 概述
本系统为所有130个二级学科生成了专业化的Agent人物画像，用于多学科协作评估研究提案的可行性和创新性。

## 文件结构
```
proposalAgent/prompts/discipline_agents/
├── index.json                    # 所有学科的索引文件
├── A01_代数与几何.txt           # 各学科的Agent prompt文件
├── A02_分析学.txt
├── ...
└── README.md                     # 本说明文件
```

## 使用方法

### 1. 基本使用
```python
from proposalAgent.agents.discipline_evaluation_integration import DisciplineEvaluationIntegration
import asyncio

async def evaluate_proposal():
    # 初始化系统
    integration = DisciplineEvaluationIntegration()
    
    # 准备研究提案数据
    proposal = {
        "title": "您的研究标题",
        "abstract": "研究摘要",
        "content": "详细研究内容",
        "keywords": ["关键词1", "关键词2"]
    }
    
    # 进行评估
    result = await integration.evaluate_proposal(proposal)
    print(result)

# 运行评估
asyncio.run(evaluate_proposal())
```

### 2. 获取特定学科的Agent prompt
```python
integration = DisciplineEvaluationIntegration()
prompt = integration.get_discipline_agent_prompt("A01")  # 获取代数与几何的prompt
print(prompt)
```

### 3. 列出所有可用学科
```python
integration = DisciplineEvaluationIntegration()
disciplines = integration.list_available_disciplines()
for disc in disciplines:
    print(f"{disc['code']} - {disc['name']} ({disc['department']})")
```

## 评估维度

每个学科Agent会从以下维度评估研究提案：

### 可行性评估
- **技术可行性**: 技术实现可能性（1-10分）
- **资源需求**: 资源需求合理性（1-10分）
- **时间框架**: 研究周期合理性（1-10分）
- **风险评估**: 技术风险和挑战（1-10分）

### 创新性评估
- **理论创新**: 理论贡献程度（1-10分）
- **方法创新**: 方法创新程度（1-10分）
- **应用创新**: 应用领域创新（1-10分）
- **交叉创新**: 跨学科创新价值（1-10分）

## 输出格式

评估结果包含：
- **overall_scores**: 综合评分（相关性、可行性、创新性）
- **recommendation**: 综合建议
- **discipline_insights**: 各学科的专业评估
- **synthesis**: 综合分析结果

## 集成到现有系统

可以将此系统集成到proposalAgent的stage2中，作为交叉性评估的一部分：

```python
# 在stage2的某个agent中
from proposalAgent.agents.discipline_evaluation_integration import DisciplineEvaluationIntegration

class CrossDisciplinaryEvaluationAgent:
    def __init__(self):
        self.discipline_evaluator = DisciplineEvaluationIntegration()
    
    async def evaluate(self, proposal_data):
        return await self.discipline_evaluator.evaluate_proposal(proposal_data)
```

## 扩展功能

1. **向量搜索集成**: 可以集成Milvus向量搜索来找到最相关的学科
2. **LLM API集成**: 可以集成真实的LLM API来获取专业评估
3. **动态协作**: 可以实现Agent之间的动态协作机制
4. **评估历史**: 可以记录和追踪评估历史

## 注意事项

- 当前系统使用模拟数据进行演示
- 实际使用时需要集成真实的LLM API
- 可以根据具体需求调整评估维度和权重
- 建议定期更新学科Agent的prompt以保持时效性
