
from datetime import date
from typing import Dict, Any, List, Optional, Union

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models.tongyi import ChatTongyi
from google import genai
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from proposalAgent.agents.utils.agent_utils import Toolkit
from proposalAgent.model_config import TONGYI_CONFIG
from proposalAgent.agents.utils.memory import EmbeddingMemory
from proposalAgent.utils.logger import get_logger
from proposalAgent.tools.academic_analysis.google_scholar import get_article_brief, resolve_author_candidates, get_author_citations, get_author_citations_auto, get_author_articles_citations
from proposalAgent.tools.academic_analysis.wos_util import wos_expanded_search, wos_expanded_citation_fanout, wos_citation_influence_summary
from proposalAgent.tools.secondary_discipline_rag import secondary_discipline_search

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector

logger = get_logger("ProposalAgentGraph")


class ProposalAgentGraph:
    """
    项目评估智能体图的主类。
    
    这个类负责：
    1. 初始化所有LLM模型和组件
    2. 创建工作流图
    3. 提供评估执行接口
    4. 支持interrupt API进行人机交互
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化项目评估图
        
        Args:
            config: 配置字典，包含LLM提供商、模型名称、API密钥等
        """
        self.config = config or TONGYI_CONFIG
        logger.info("初始化ProposalAgentGraph，使用LLM提供商: %s", self.config['llm_provider'])
        
        # 初始化LLM模型
        self._initialize_llms()
        
        # 初始化工具包
        self.toolkit = Toolkit(config=self.config)
        
        # 初始化记忆系统
        self._initialize_memories()
        
        # 创建工具节点
        self.tool_nodes = self._create_tool_nodes()
        
        # 初始化条件逻辑
        self.conditional_logic = ConditionalLogic()
        
        # 创建图
        self.graph_setup = GraphSetup(
            quick_thinking_llm=self.quick_thinking_llm,
            deep_think_llm=self.deep_thinking_llm,
            structure_llm=self.structure_llm,
            tool_nodes=self.tool_nodes,
            toolkit=self.toolkit,
            conditional_logic=self.conditional_logic,
            feasible_good_memory=self.feasible_good_memory,
            feasible_bad_memory=self.feasible_bad_memory,
            feasible_manager_memory=self.feasible_manager_memory,
            innovation_good_memory=self.innovation_good_memory,
            innovation_bad_memory=self.innovation_bad_memory,
            innovation_manager_memory=self.innovation_manager_memory,
        )
        
        # 当前状态和反思器
        self.curr_state = None
        self.reflector = Reflector(self.quick_thinking_llm)
        
        # 创建工作流图（包含interrupt支持）
        self.graph = self.graph_setup.setup_graph()
        
        # 传播器用于创建初始状态
        self.propagator = Propagator()
        
        logger.info("ProposalAgentGraph初始化完成")
    
    def _initialize_llms(self):
        """初始化LLM模型"""
        provider = self.config["llm_provider"].lower()
        
        if provider in ["openai", "ollama", "openrouter"]:
            self.deep_thinking_llm = ChatOpenAI(
                model=self.config["deep_think_llm"], 
                base_url=self.config["backend_url"]
            )
            self.quick_thinking_llm = ChatOpenAI(
                model=self.config["quick_think_llm"], 
                base_url=self.config["backend_url"]
            )
        elif provider == "tongyi":
            self.deep_thinking_llm = ChatTongyi(
                model=self.config["deep_think_llm"], 
                api_key=self.config["api_key"]
            )
            self.quick_thinking_llm = ChatTongyi(
                model=self.config["quick_think_llm"], 
                api_key=self.config["api_key"]
            )
        elif provider == "google":
            self.deep_thinking_llm = ChatGoogleGenerativeAI(
                model=self.config["deep_think_llm"]
            )
            self.quick_thinking_llm = ChatGoogleGenerativeAI(
                model=self.config["quick_think_llm"]
            )
        else:
            raise ValueError(f"不支持的LLM提供商: {self.config['llm_provider']}")
        
        # 结构化LLM（用于特殊任务）
        try:
            self.structure_llm = genai.Client().models
        except Exception as e:
            logger.warning("无法初始化Google Genai客户端: %s", e)
            self.structure_llm = None
    
    def _initialize_memories(self):
        """初始化各种记忆系统"""
        # self.planning_memory = EmbeddingMemory(name="planning_analysis", config=self.config)
        # self.impact_memory = EmbeddingMemory(name="impact_analysis", config=self.config)
        # self.future_influence_memory = EmbeddingMemory(name="future_influence_memory", config=self.config)
        # self.risk_memory = EmbeddingMemory(name="risk_memory", config=self.config)
        # self.interdisciplinary_memory = EmbeddingMemory(name="interdisciplinary_memory", config=self.config)
        # self.academic_memory = EmbeddingMemory(name="academic_memory", config=self.config)
        # self.feasibility_memory = EmbeddingMemory(name="feasibility_memory", config=self.config)
        # self.innovation_memory = EmbeddingMemory(name="innovation_memory", config=self.config)
        
        self.feasible_good_memory = EmbeddingMemory(name="feasible_good_memory", config=self.config)
        self.feasible_bad_memory = EmbeddingMemory(name="feasible_bad_memory", config=self.config)
        self.feasible_manager_memory = EmbeddingMemory(name="feasible_manager_memory", config=self.config)
        
        self.innovation_good_memory = EmbeddingMemory(name="innovation_good_memory", config=self.config)
        self.innovation_bad_memory = EmbeddingMemory(name="innovation_bad_memory", config=self.config)
        self.innovation_manager_memory = EmbeddingMemory(name="innovation_manager_memory", config=self.config)
        
        # self.final_analyst_memory = EmbeddingMemory(name="final_analyst_memory", config=self.config)
        # self.feedback_analysis_memory = EmbeddingMemory(name="feedback_analysis_memory", config=self.config)
        # self.completeness_checker_memory = EmbeddingMemory(name="completeness_checker_memory", config=self.config)
        # self.generator_memory = EmbeddingMemory(name="generator_memory", config=self.config)
        

    
    def _create_tool_nodes(self):
        """创建工具节点"""
        return {
            "academic": ToolNode([
                get_article_brief, resolve_author_candidates, get_author_citations, get_author_citations_auto, get_author_articles_citations,
          #      wos_expanded_search, wos_expanded_citation_fanout, wos_citation_influence_summary
                ]),
            "social": ToolNode([]),
            "influence": ToolNode([
                ]),
            "interdisciplinary": ToolNode([secondary_discipline_search]),
            "feasibility": ToolNode([]),
            "innovation": ToolNode([]),
        }
    
    def evaluate_project(
        self, 
        user_prompt: str, 
        user_interests: Optional[List[str]] = None, 
        filepath: str = "",
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行项目评估，支持interrupt API进行人机交互
        
        Args:
            user_prompt: 用户的项目评估请求
            user_interests: 用户关注的评估重点列表
            filepath: 项目文档路径
            thread_id: 线程ID，用于支持interrupt恢复
            
        Returns:
            Dict包含评估结果，如果发生中断则包含中断信息
        """
        if user_interests is None:
            user_interests = []
            
        if thread_id is None:
            thread_id = f"evaluation_{hash(user_prompt)}_{date.today().isoformat()}"
        
        # 创建初始状态
        initial_state = self.propagator.create_initial_state(
            user_prompt=user_prompt,
            user_interest=user_interests,
            filepath=filepath
        )
        
        # 配置线程
        thread_config = {"configurable": {"thread_id": thread_id}}
        
        logger.info("开始项目评估，线程ID: %s", thread_id)
        
        try:
            # 执行图
            result = self.graph.invoke(initial_state, config=thread_config)
            
            # 检查是否有中断
            if "__interrupt__" in result and result["__interrupt__"]:
                logger.info("检测到人类审核中断")
                interrupt_info = result["__interrupt__"][0]
                
                return {
                    "status": "interrupted",
                    "thread_id": thread_id,
                    "interrupt_info": interrupt_info.value,
                    "message": "评估流程已暂停，等待人类审核",
                    "partial_result": {
                        key: value for key, value in result.items() 
                        if not key.startswith("__")
                    }
                }
            else:
                logger.info("项目评估完成")
                self.curr_state = result
                
                return {
                    "status": "completed",
                    "thread_id": thread_id,
                    "final_report": result.get("final_report", ""),
                    "analysis_summary": result.get("final_analysis_summary", ""),
                    "academic_analysis": result.get("academic_analysis_report", ""),
                    "social_analysis": result.get("social_analysis_report", ""),
                    "future_influence": result.get("future_influence_report", ""),
                    "debate_results": result.get("debate_results", {}),
                    "full_result": result
                }
                
        except Exception as e:
            logger.error("项目评估过程中发生错误: %s", e)
            return {
                "status": "error",
                "thread_id": thread_id,
                "error": str(e),
                "message": "评估过程中发生错误"
            }
    
    def resume_evaluation(
        self, 
        thread_id: str, 
        human_feedback: str
    ) -> Dict[str, Any]:
        """
        使用人类反馈恢复评估
        
        Args:
            thread_id: 要恢复的线程ID
            human_feedback: 人类提供的反馈
            
        Returns:
            Dict包含恢复后的评估结果
        """
        thread_config = {"configurable": {"thread_id": thread_id}}
        
        logger.info("恢复评估，线程ID: %s", thread_id)
        logger.info("人类反馈: %s...", human_feedback[:100])
        
        try:
            # 使用Command.resume恢复执行
            result = self.graph.invoke(
                Command(resume={"feedback": human_feedback}),
                config=thread_config
            )
            
            # 检查是否再次中断
            if "__interrupt__" in result and result["__interrupt__"]:
                logger.info("评估再次中断")
                interrupt_info = result["__interrupt__"][0]
                
                return {
                    "status": "interrupted",
                    "thread_id": thread_id,
                    "interrupt_info": interrupt_info.value,
                    "message": "评估流程再次暂停，等待进一步审核",
                    "partial_result": {
                        key: value for key, value in result.items() 
                        if not key.startswith("__")
                    }
                }
            else:
                logger.info("评估恢复并完成")
                self.curr_state = result
                
                return {
                    "status": "completed",
                    "thread_id": thread_id,
                    "final_report": result.get("final_report", ""),
                    "analysis_summary": result.get("final_analysis_summary", ""),
                    "academic_analysis": result.get("academic_analysis_report", ""),
                    "social_analysis": result.get("social_analysis_report", ""),
                    "future_influence": result.get("future_influence_report", ""),
                    "debate_results": result.get("debate_results", {}),
                    "human_feedback": human_feedback,
                    "full_result": result
                }
                
        except Exception as e:
            logger.error("恢复评估过程中发生错误: %s", e)
            return {
                "status": "error",
                "thread_id": thread_id,
                "error": str(e),
                "message": "恢复评估过程中发生错误"
            }
    
    def get_evaluation_status(self, thread_id: str) -> Dict[str, Any]:
        """
        获取评估状态
        
        Args:
            thread_id: 线程ID
            
        Returns:
            Dict包含评估状态信息
        """
        # 这里可以添加获取状态的逻辑
        # 由于LangGraph的checkpointer机制，我们可以查询特定线程的状态
        return {
            "thread_id": thread_id,
            "message": "状态查询功能待实现"
        }
    
    def reflect_and_remember(self, evaluation_outcome: Dict[str, Any]):
        """
        基于评估结果进行反思并更新记忆
        
        Args:
            evaluation_outcome: 评估结果字典
        """
        if not self.curr_state:
            logger.warning("没有当前状态可以反思")
            return
            
        logger.info("开始反思和记忆更新")
        
        # 反思各个分析组件
        try:
            self.reflector.reflect_academic_analyst(
                self.curr_state, evaluation_outcome, self.academic_memory
            )
            self.reflector.reflect_future_influence_analyst(
                self.curr_state, evaluation_outcome, self.future_influence_memory
            )
            self.reflector.reflect_interdisciplinary_analyst(
                self.curr_state, evaluation_outcome, self.interdisciplinary_memory
            )
            self.reflector.reflect_feasibility_debate(
                self.curr_state, evaluation_outcome, self.feasibility_memory
            )
            self.reflector.reflect_innovation_debate(
                self.curr_state, evaluation_outcome, self.innovation_memory
            )
            
            logger.info("反思和记忆更新完成")
        except Exception as e:
            logger.error("反思过程中发生错误: %s", e)
    
    def get_graph_info(self) -> Dict[str, Any]:
        """
        获取图的基本信息
        
        Returns:
            Dict包含图的配置和状态信息
        """
        return {
            "llm_provider": self.config["llm_provider"],
            "deep_think_model": self.config["deep_think_llm"],
            "quick_think_model": self.config["quick_think_llm"],
            "graph_nodes": len(self.graph.nodes) if hasattr(self.graph, 'nodes') else "未知",
            "interrupt_supported": True,
            "checkpointer_enabled": True
        }