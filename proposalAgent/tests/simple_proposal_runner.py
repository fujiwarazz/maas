#!/usr/bin/env python3
"""
简化的项目评估运行器
提供简单的graph.invoke()接口，可以直接运行完整的评估流程
"""

from typing import Dict, Any, Optional, List
from proposalAgent.graphs.simple_workflow import SimpleWorkflow
from proposalAgent.model_config import TONGYI_CONFIG
from proposalAgent.utils.logger import get_logger

logger = get_logger("SimpleProposalRunner")


class SimpleProposalRunner:
    """
    简化的项目评估运行器
    
    使用方法:
        runner = SimpleProposalRunner()
        result = runner.graph.invoke({"messages": [("user", "请分析这个项目")]})
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化简化的项目评估运行器
        
        Args:
            config: 配置字典，如果不提供则使用默认配置
        """
        self.config = config or TONGYI_CONFIG
        logger.info("初始化SimpleProposalRunner")
        
        # 创建简化工作流
        self.workflow = SimpleWorkflow(config=self.config)
        
        # 提供graph属性，方便用户调用
        self.graph = self.workflow.graph
        
        logger.info("SimpleProposalRunner初始化完成")
    
    def evaluate(
        self, 
        user_prompt: str, 
        filepath: str = "",
        user_interests: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        执行项目评估
        
        Args:
            user_prompt: 用户的评估请求
            filepath: 文件路径（可选）
            user_interests: 用户关注点（可选）
            
        Returns:
            Dict包含评估结果
        """
        return self.workflow.run_evaluation(
            user_prompt=user_prompt,
            filepath=filepath,
            user_interests=user_interests
        )
    
    def get_info(self) -> Dict[str, Any]:
        """获取工作流信息"""
        return self.workflow.get_workflow_info()


def create_simple_runner(config: Optional[Dict[str, Any]] = None) -> SimpleProposalRunner:
    """
    便捷函数：创建简化的项目评估运行器
    
    Args:
        config: 可选配置
        
    Returns:
        SimpleProposalRunner实例
    """
    return SimpleProposalRunner(config=config)


# 示例使用
if __name__ == "__main__":
    # 创建运行器
    runner = create_simple_runner()
    
    # 示例1: 直接使用graph.invoke()
    print("=== 示例1: 直接使用graph.invoke() ===")
    try:
        result = runner.graph.invoke({
            "messages": [("user", "请分析人工智能在医疗领域的应用前景")],
            "research_topic": ["人工智能", "医疗"],
            "filepath": ""
        })
        print("评估完成!")
        print(f"最终报告: {result.get('final_report', 'N/A')[:200]}...")
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
    
    # 示例2: 使用便捷方法
    print("\n=== 示例2: 使用便捷方法 ===")
    try:
        result = runner.evaluate(
            user_prompt="请评估区块链技术在金融行业的可行性",
            user_interests=["技术可行性", "市场前景", "风险评估"]
        )
        print("评估完成!")
        print(f"状态: {result.get('status')}")
        print(f"学术分析: {result.get('academic_analysis', 'N/A')[:100]}...")
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
    
    # 获取工作流信息
    print("\n=== 工作流信息 ===")
    info = runner.get_info()
    for key, value in info.items():
        print(f"{key}: {value}")
