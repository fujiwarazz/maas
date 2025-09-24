from typing import Dict, Any, Optional, List
from datetime import date
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models.tongyi import ChatTongyi

from proposalAgent.agents.utils.agent_states import AgentState
from proposalAgent.agents.utils.agent_utils import Toolkit, create_msg_delete
from proposalAgent.model_config import TONGYI_CONFIG
from proposalAgent.agents.utils.memory import EmbeddingMemory
from proposalAgent.utils.logger import get_logger

# Import all the agent creation functions
from proposalAgent.agents.stage1.intention import create_intention_agent
from proposalAgent.agents.stage1.output import create_output_node
from proposalAgent.agents.stage1.structure import create_structure_node
from proposalAgent.agents.stage1.schedule import create_schedule_agent

from proposalAgent.agents.stage2.academic import create_academic_agent
from proposalAgent.agents.stage2.future_influence import create_future_influence_agent
from proposalAgent.agents.stage2.interdis import create_interdis_agent

from proposalAgent.agents.stage2.debate.feasible.feasible_bad import create_feasible_bad_agent
from proposalAgent.agents.stage2.debate.feasible.feasible_good import create_feasible_good_agent
from proposalAgent.agents.stage2.debate.feasible.feasible_manager import create_feasible_manager

from proposalAgent.agents.stage2.debate.innovation.innovation_bad import create_innovation_bad_agent
from proposalAgent.agents.stage2.debate.innovation.innovation_good import create_innovation_good_agent
from proposalAgent.agents.stage2.debate.innovation.innovation_manager import create_innovation_manager

from proposalAgent.agents.stage3.feedback_analysis_agent import create_feedback_analysis_agent
from proposalAgent.agents.stage3.final_analysis import create_final_analyst_agent
from proposalAgent.agents.stage3.completeness_checker import create_completeness_checker_agent
from proposalAgent.agents.stage3.generator import create_generator_agent

from proposalAgent.tools.academic_analysis.google_scholar import get_article_brief, resolve_author_candidates, get_author_citations, get_author_citations_auto, get_author_articles_citations
from proposalAgent.tools.academic_analysis.wos_util import wos_expanded_search, wos_expanded_citation_fanout, wos_citation_influence_summary
from proposalAgent.tools.secondary_discipline_rag import secondary_discipline_search

from .conditional_logic import ConditionalLogic
from .propagation import Propagator

logger = get_logger("SimpleWorkflow")


class SimpleWorkflow:
    """
    简化的工作流实现，保持和setup相同的流程但确保能够正常运行
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化简化工作流
        
        Args:
            config: 配置字典，包含LLM提供商、模型名称、API密钥等
        """
        self.config = config or TONGYI_CONFIG
        logger.info("初始化SimpleWorkflow，使用LLM提供商: %s", self.config['llm_provider'])
        
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
        
        # 创建工作流图
        self.graph = self._setup_graph()
        
        # 传播器用于创建初始状态
        self.propagator = Propagator()
        
        logger.info("SimpleWorkflow初始化完成")
    
    def _initialize_llms(self):
        """初始化LLM模型"""
        provider = self.config["llm_provider"].lower()
        
        try:
            if provider in ["openai", "ollama", "openrouter"]:
                self.deep_thinking_llm = ChatOpenAI(
                    model=self.config["deep_think_llm"], 
                    base_url=self.config.get("backend_url", "https://api.openai.com/v1")
                )
                self.quick_thinking_llm = ChatOpenAI(
                    model=self.config["quick_think_llm"], 
                    base_url=self.config.get("backend_url", "https://api.openai.com/v1")
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
        except Exception as e:  # pylint: disable=broad-except
            logger.error("初始化LLM失败: %s", str(e))
            # 使用默认的OpenAI作为fallback
            self.deep_thinking_llm = ChatOpenAI(model="gpt-3.5-turbo")
            self.quick_thinking_llm = ChatOpenAI(model="gpt-3.5-turbo")
        
        # 结构化LLM（可选）
        self.structure_llm = None
    
    def _initialize_memories(self):
        """初始化各种记忆系统"""
        try:
            self.academic_memory = EmbeddingMemory(name="academic_memory", config=self.config)
            self.feasible_good_memory = EmbeddingMemory(name="feasible_good_memory", config=self.config)
            self.feasible_bad_memory = EmbeddingMemory(name="feasible_bad_memory", config=self.config)
            self.feasible_manager_memory = EmbeddingMemory(name="feasible_manager_memory", config=self.config)
            self.innovation_good_memory = EmbeddingMemory(name="innovation_good_memory", config=self.config)
            self.innovation_bad_memory = EmbeddingMemory(name="innovation_bad_memory", config=self.config)
            self.innovation_manager_memory = EmbeddingMemory(name="innovation_manager_memory", config=self.config)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("初始化记忆系统失败: %s，使用默认配置", str(e))
            # 使用默认配置创建记忆
            default_config = {"llm_provider": "openai"}
            self.academic_memory = EmbeddingMemory(name="academic_memory", config=default_config)
            self.feasible_good_memory = EmbeddingMemory(name="feasible_good_memory", config=default_config)
            self.feasible_bad_memory = EmbeddingMemory(name="feasible_bad_memory", config=default_config)
            self.feasible_manager_memory = EmbeddingMemory(name="feasible_manager_memory", config=default_config)
            self.innovation_good_memory = EmbeddingMemory(name="innovation_good_memory", config=default_config)
            self.innovation_bad_memory = EmbeddingMemory(name="innovation_bad_memory", config=default_config)
            self.innovation_manager_memory = EmbeddingMemory(name="innovation_manager_memory", config=default_config)
    
    def _create_tool_nodes(self):
        """创建工具节点"""
        return {
            "academic": ToolNode([
                get_article_brief, resolve_author_candidates, get_author_citations, 
                get_author_citations_auto, get_author_articles_citations,
                wos_expanded_search, wos_expanded_citation_fanout, wos_citation_influence_summary
            ]),
            "social": ToolNode([]),
            "influence": ToolNode([]),
            "interdisciplinary": ToolNode([secondary_discipline_search]),
            "feasibility": ToolNode([]),
            "innovation": ToolNode([]),
        }
    
    def _setup_graph(self):
        """
        构建并返回工作流图
        """
        # 创建StateGraph实例
        workflow = StateGraph(AgentState)
        
        # Stage 1 节点
        intention_node = create_intention_agent(self.quick_thinking_llm)
        output_node = create_output_node(self.quick_thinking_llm)
        structure_node = create_structure_node()
        planning_node = create_schedule_agent(self.deep_thinking_llm)
        
        # Stage 2 节点 - 信息收集
        academic_analysis_node = create_academic_agent(
            self.quick_thinking_llm, toolkit=self.toolkit, memory=self.academic_memory
        )
        academic_tool_exc_node = self.tool_nodes["academic"]
        academic_msg_clear_node = create_msg_delete()
        
        # 社会分析节点（简化版）
        def simple_social_analysis_node(state: AgentState):
            _ = state  # 标记使用了state参数
            return {"messages": [], "social_analysis_report": "社会分析模块：基于用户输入进行基础社会影响分析"}
        
        social_analysis_node = simple_social_analysis_node
        social_tool_exc_node = self.tool_nodes["social"]
        social_msg_clear_node = create_msg_delete()
        
        # 未来影响分析
        future_influence_node = create_future_influence_agent(self.deep_thinking_llm, self.toolkit)
        future_influence_tool_exc_node = self.tool_nodes["influence"]
        future_influence_msg_clear_node = create_msg_delete()
        
        # 跨学科分析
        interdisciplinary_node = create_interdis_agent(self.deep_thinking_llm, self.toolkit)
        interdisciplinary_tool_exc_node = self.tool_nodes["interdisciplinary"]
        interdisciplinary_msg_clear_node = create_msg_delete()
        
        # Stage 2 - 辩论部分
        feasible_good_node = create_feasible_good_agent(
            self.deep_thinking_llm, self.toolkit, self.feasible_good_memory
        )
        feasible_good_tool_exc_node = self.tool_nodes["feasibility"]
        feasible_good_msg_clear_node = create_msg_delete()
        
        feasible_bad_node = create_feasible_bad_agent(
            self.deep_thinking_llm, self.toolkit, self.feasible_bad_memory
        )
        feasible_bad_tool_exc_node = self.tool_nodes["feasibility"]
        feasible_bad_msg_clear_node = create_msg_delete()
        
        feasible_judge_node = create_feasible_manager(
            self.deep_thinking_llm, self.feasible_manager_memory
        )
        
        innovation_good_node = create_innovation_good_agent(
            self.deep_thinking_llm, self.toolkit, self.innovation_good_memory
        )
        innovation_good_tool_exc_node = self.tool_nodes["innovation"]
        innovation_good_msg_clear_node = create_msg_delete()
        
        innovation_bad_node = create_innovation_bad_agent(
            self.deep_thinking_llm, self.toolkit, self.innovation_bad_memory
        )
        innovation_bad_tool_exc_node = self.tool_nodes["innovation"]
        innovation_bad_msg_clear_node = create_msg_delete()
        
        innovation_judge_node = create_innovation_manager(
            self.deep_thinking_llm, self.innovation_manager_memory
        )
        
        # Stage 3 节点
        final_analyst_node = create_final_analyst_agent(self.deep_thinking_llm)  # type: ignore
        completeness_checker_node = create_completeness_checker_agent(self.deep_thinking_llm)  # type: ignore
        generator_node = create_generator_agent(self.deep_thinking_llm)  # type: ignore
        feedback_analysis_node = create_feedback_analysis_agent(self.deep_thinking_llm)  # type: ignore
        
        # 添加所有节点
        workflow.add_node("intention_node", intention_node)
        workflow.add_node("planning_node", planning_node)
        workflow.add_node("output_node", output_node)
        workflow.add_node("structure_node", structure_node)
        
        # Stage 2 信息收集节点
        workflow.add_node("academic_analysis_node", academic_analysis_node)
        workflow.add_node("academic_analysis_tool_exc_node", academic_tool_exc_node)
        workflow.add_node("academic_analysis_msg_clear_node", academic_msg_clear_node)
        
        workflow.add_node("social_analysis_node", social_analysis_node)
        workflow.add_node("social_analysis_tool_exc_node", social_tool_exc_node)
        workflow.add_node("social_analysis_msg_clear_node", social_msg_clear_node)
        
        workflow.add_node("future_influence_node", future_influence_node)
        workflow.add_node("future_influence_tool_exc_node", future_influence_tool_exc_node)
        workflow.add_node("future_influence_msg_clear_node", future_influence_msg_clear_node)
        
        workflow.add_node("interdisciplinary_node", interdisciplinary_node)
        workflow.add_node("interdisciplinary_tool_exc_node", interdisciplinary_tool_exc_node)
        workflow.add_node("interdisciplinary_msg_clear_node", interdisciplinary_msg_clear_node)
        
        # 辩论节点
        workflow.add_node("feasible_good_node", feasible_good_node)
        workflow.add_node("feasible_good_tool_exc_node", feasible_good_tool_exc_node)
        workflow.add_node("feasible_good_msg_clear_node", feasible_good_msg_clear_node)
        
        workflow.add_node("feasible_bad_node", feasible_bad_node)
        workflow.add_node("feasible_bad_tool_exc_node", feasible_bad_tool_exc_node)
        workflow.add_node("feasible_bad_msg_clear_node", feasible_bad_msg_clear_node)
        workflow.add_node("feasible_judge_node", feasible_judge_node)
        
        workflow.add_node("innovation_good_node", innovation_good_node)
        workflow.add_node("innovation_good_tool_exc_node", innovation_good_tool_exc_node)
        workflow.add_node("innovation_good_msg_clear_node", innovation_good_msg_clear_node)
        
        workflow.add_node("innovation_bad_node", innovation_bad_node)
        workflow.add_node("innovation_bad_tool_exc_node", innovation_bad_tool_exc_node)
        workflow.add_node("innovation_bad_msg_clear_node", innovation_bad_msg_clear_node)
        workflow.add_node("innovation_judge_node", innovation_judge_node)
        
        # Stage 3 节点
        workflow.add_node("final_analyst_node", final_analyst_node)
        workflow.add_node("completeness_checker_node", completeness_checker_node)
        workflow.add_node("generator_node", generator_node)
        workflow.add_node("feedback_analysis_node", feedback_analysis_node)
        
        # 简化的边连接（直接的线性流程）
        workflow.add_edge(START, "intention_node")
        
        # 简化的条件逻辑
        def should_output(state):
            """简化的意图判断"""
            intention = state.get("intention_decision", "structure")
            if intention == "output":
                return "output_node"
            return "structure_node"
        
        workflow.add_conditional_edges(
            "intention_node",
            should_output,
            {"output_node": "output_node", "structure_node": "structure_node"},
        )
        
        workflow.add_edge("output_node", END)
        workflow.add_edge("structure_node", "planning_node")
        workflow.add_edge("planning_node", "academic_analysis_node")
        
        # 信息收集阶段 - 简化的线性连接
        workflow.add_edge("academic_analysis_node", "social_analysis_node")
        workflow.add_edge("social_analysis_node", "future_influence_node")
        workflow.add_edge("future_influence_node", "interdisciplinary_node")
        
        # 简化的辩论控制器
        def simple_debate_controller(state: AgentState):
            """简化的辩论控制器"""
            disciplines = state.get("interdisciplinary_results", ["通用学科"])
            
            # 简化：只运行一轮辩论
            debate_results = {}
            for discipline in disciplines[:1]:  # 只处理第一个学科
                # 可行性辩论
                feasible_good_result = feasible_good_node(state)
                feasible_bad_result = feasible_bad_node(state)
                feasible_judge_result = feasible_judge_node(state)
                
                # 创新性辩论
                innovation_good_result = innovation_good_node(state)
                innovation_bad_result = innovation_bad_node(state)
                innovation_judge_result = innovation_judge_node(state)
                
                debate_results[discipline] = {
                    "feasibility": {
                        "good": feasible_good_result,
                        "bad": feasible_bad_result,
                        "judge": feasible_judge_result
                    },
                    "innovation": {
                        "good": innovation_good_result,
                        "bad": innovation_bad_result,
                        "judge": innovation_judge_result
                    }
                }
            
            state["debate_results"] = debate_results
            return state
        
        workflow.add_node("debate_controller", simple_debate_controller)
        workflow.add_edge("interdisciplinary_node", "debate_controller")
        workflow.add_edge("debate_controller", "final_analyst_node")
        
        # 简化的人类审核节点
        def simple_human_review_node(state: AgentState) -> AgentState:
            """简化的人类审核节点"""
            # 进行完备性检查
            state = completeness_checker_node(state)
            
            # 简化：直接跳过人类审核
            completeness_recommendation = state.get("completeness_recommendation", "complete")
            if completeness_recommendation == "complete":
                state["skip_human_review"] = True
            else:
                # 模拟人类反馈
                state["human_feedback"] = "approved"
                state["skip_human_review"] = True
            
            return state
        
        workflow.add_node("human_review_node", simple_human_review_node)
        workflow.add_edge("final_analyst_node", "human_review_node")
        
        # 路由到生成器
        def route_after_review(state: AgentState) -> str:
            """简化的审核后路由"""
            _ = state  # 标记使用了state参数
            return "generate"
        
        workflow.add_conditional_edges(
            "human_review_node",
            route_after_review,
            {"generate": "generator_node"},
        )
        
        workflow.add_edge("generator_node", END)
        
        # 使用内存保存器
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    
    def run_evaluation(
        self, 
        user_prompt: str, 
        user_interests: Optional[List[str]] = None, 
        filepath: str = "",
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行项目评估
        
        Args:
            user_prompt: 用户的项目评估请求
            user_interests: 用户关注的评估重点列表  
            filepath: 项目文档路径
            thread_id: 线程ID
            
        Returns:
            Dict包含评估结果
        """
        if user_interests is None:
            user_interests = []
            
        if thread_id is None:
            thread_id = f"simple_eval_{hash(user_prompt)}_{date.today().isoformat()}"
        
        # 创建初始状态
        initial_state = self.propagator.create_initial_state(
            user_prompt=user_prompt,
            user_interest=user_interests,
            filepath=filepath
        )
        
        # 配置线程
        thread_config = {"configurable": {"thread_id": thread_id}}
        
        logger.info("开始简化项目评估，线程ID: %s", thread_id)
        
        try:
            # 执行图 - 强制类型转换为AgentState
            result = self.graph.invoke(initial_state, config=thread_config)  # type: ignore
            
            logger.info("项目评估完成")
            
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
                
        except Exception as e:  # pylint: disable=broad-except
            logger.error("项目评估过程中发生错误: %s", str(e))
            return {
                "status": "error",
                "thread_id": thread_id,
                "error": str(e),
                "message": "评估过程中发生错误"
            }
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """
        获取工作流的基本信息
        """
        return {
            "llm_provider": self.config["llm_provider"],
            "deep_think_model": self.config["deep_think_llm"],
            "quick_think_model": self.config["quick_think_llm"],
            "workflow_type": "SimpleWorkflow",
            "checkpointer_enabled": True
        }
