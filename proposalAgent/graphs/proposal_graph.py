
import asyncio
import uuid
from datetime import date
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Tuple, Union

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
# from proposalAgent.tools.academic_analysis.wos_util import wos_expanded_search, wos_expanded_citation_fanout, wos_citation_influence_summary
from proposalAgent.tools.secondary_discipline_rag import secondary_discipline_search
from proposalAgent.tools.baidu_util import baidu_search_with_content
from proposalAgent.tools.tavily_util import tavily_search
from proposalAgent.graphs.setup import GraphSetup
from proposalAgent.graphs.conditional_logic import ConditionalLogic
from proposalAgent.graphs.propagation import Propagator
from proposalAgent.graphs.reflection import Reflector

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
        
        self._initialize_llms()
        
        self.toolkit = Toolkit(config=self.config)
        
        self._initialize_memories()
        
        self.tool_nodes = self._create_tool_nodes()
        
        self.conditional_logic = ConditionalLogic()
        if getattr(self.conditional_logic, "max_debate_rounds", 1) < 2:
            self.conditional_logic.max_debate_rounds = 2
        
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
                get_article_brief, resolve_author_candidates, get_author_citations, get_author_citations_auto, get_author_articles_citations,tavily_search
          #      wos_expanded_search, wos_expanded_citation_fanout, wos_citation_influence_summary
                ]),
            "social": ToolNode([]),
            "influence": ToolNode([
                tavily_search
                ]),
            "interdisciplinary": ToolNode([secondary_discipline_search]),
            "feasibility": ToolNode([]),
            "innovation": ToolNode([]),
        }
    
    @staticmethod
    def _run_sync(awaitable: asyncio.coroutines.iscoroutine) -> Any:
        try:
            return asyncio.run(awaitable)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(awaitable)
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    def _build_thread_config(
        self, thread_id: Optional[str] = None, recursion_limit: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        
        if thread_id is None:
            thread_id = f"evaluation_{uuid.uuid4().hex}"
        config: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        limit = recursion_limit or self.propagator.max_recur_limit
        config["config"] = {"recursion_limit": limit}
        return thread_id, config

    def create_session(
        self, thread_id: Optional[str] = None, recursion_limit: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        return self._build_thread_config(thread_id, recursion_limit)

    async def stream_project(
        self,
        initial_state: Dict[str, Any],
        thread_config: Dict[str, Any],
        feedback_handler: Optional[Callable[[Any], Awaitable[Optional[str]]]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        payload: Union[Dict[str, Any], Command] = initial_state
        final_state: Optional[Dict[str, Any]] = None
       
        while True:
            stream = self.graph.astream(payload, config=thread_config)
            async for chunk in stream:
                if cancel_event and cancel_event.is_set():
                    self.curr_state = final_state
                    return

                if isinstance(chunk, dict):
                    final_state = chunk

                interrupt = chunk.get("__interrupt__") if isinstance(chunk, dict) else None
                if interrupt:
                    yield {"__interrupt__": interrupt, "state": final_state}

                    if feedback_handler is None:
                        self.curr_state = final_state
                        return

                    feedback = await feedback_handler(interrupt[0])
                    if feedback is None:
                        self.curr_state = final_state
                        return

                    payload = Command(resume={"feedback": feedback})
                    break
            else:
                self.curr_state = final_state
                if final_state is not None:
                    yield {"state": final_state}
                return

    async def evaluate_project(
        self,
        user_prompt: str,
        user_interests: Optional[List[str]] = None,
        filepath: str = "",
        thread_id: Optional[str] = None,
        feedback_handler: Optional[Callable[[Any], Awaitable[Optional[str]]]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        if user_interests is None:
            user_interests = []

        thread_id, thread_config = self._build_thread_config(thread_id)

        initial_state = self.propagator.create_initial_state(
            user_prompt=user_prompt,
            user_interest=user_interests,
            filepath=filepath,
        )

        logger.info("开始项目评估，线程ID: %s", thread_id)

        try:
            result = await self.graph.ainvoke(initial_state, config=thread_config)

            while "__interrupt__" in result and result["__interrupt__"]:
                if cancel_event and cancel_event.is_set():
                    logger.info("接收到取消信号，终止评估 (thread_id=%s)", thread_id)
                    return {"status": "cancelled", "thread_id": thread_id}

                interrupt_payload = result["__interrupt__"][0]
                payload_value = getattr(interrupt_payload, "value", interrupt_payload)

                if feedback_handler:
                    human_input = await feedback_handler(payload_value)
                    if human_input is None:
                        logger.info("反馈处理器返回 None，终止评估 (thread_id=%s)", thread_id)
                        return {"status": "cancelled", "thread_id": thread_id}
                else:
                    print("🛑 工作流等待人类反馈：")
                    print(payload_value)
                    human_input = input("请输入人类反馈（例如: approved）: ")

                result = await self.graph.ainvoke(
                    Command(resume={"feedback": human_input}),
                    config=thread_config,
                )

            if cancel_event and cancel_event.is_set():
                logger.info("接收到取消信号，终止评估 (thread_id=%s)", thread_id)
                return {"status": "cancelled", "thread_id": thread_id}

            self.curr_state = result

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
                "full_result": result,
            }
        except Exception as e:
            logger.error("项目评估过程中发生错误: %s", e)
            return {
                "status": "error",
                "thread_id": thread_id,
                "error": str(e),
                "message": "评估过程中发生错误",
            }

    def evaluate_project_sync(
        self,
        user_prompt: str,
        user_interests: Optional[List[str]] = None,
        filepath: str = "",
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._run_sync(
            self.evaluate_project(
                user_prompt=user_prompt,
                user_interests=user_interests,
                filepath=filepath,
                thread_id=thread_id,
                **kwargs,
            )
        )

    async def resume_evaluation(
        self,
        thread_id: str,
        human_feedback: str,
        feedback_handler: Optional[Callable[[Any], Awaitable[Optional[str]]]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        thread_id, thread_config = self._build_thread_config(thread_id)

        logger.info("恢复评估，线程ID: %s", thread_id)
        logger.info("人类反馈: %s...", human_feedback[:100])

        try:
            result = await self.graph.ainvoke(
                Command(resume={"feedback": human_feedback}),
                config=thread_config,
            )

            while "__interrupt__" in result and result["__interrupt__"]:
                if cancel_event and cancel_event.is_set():
                    logger.info("恢复流程收到取消信号 (thread_id=%s)", thread_id)
                    return {"status": "cancelled", "thread_id": thread_id}

                interrupt_payload = result["__interrupt__"][0]
                payload_value = getattr(interrupt_payload, "value", interrupt_payload)

                if feedback_handler:
                    human_input = await feedback_handler(payload_value)
                    if human_input is None:
                        logger.info("恢复流程中反馈处理器返回 None，终止")
                        return {"status": "cancelled", "thread_id": thread_id}
                else:
                    print("🛑 工作流再次等待反馈：")
                    print(payload_value)
                    human_input = input("请输入新的反馈: ")

                result = await self.graph.ainvoke(
                    Command(resume={"feedback": human_input}),
                    config=thread_config,
                )

            if cancel_event and cancel_event.is_set():
                logger.info("恢复流程收到取消信号 (thread_id=%s)", thread_id)
                return {"status": "cancelled", "thread_id": thread_id}

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
                "full_result": result,
            }
        except Exception as e:
            logger.error("恢复评估过程中发生错误: %s", e)
            return {
                "status": "error",
                "thread_id": thread_id,
                "error": str(e),
                "message": "恢复评估过程中发生错误",
            }

    def resume_evaluation_sync(
        self,
        thread_id: str,
        human_feedback: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._run_sync(
            self.resume_evaluation(
                thread_id=thread_id,
                human_feedback=human_feedback,
                **kwargs,
            )
        )
    
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