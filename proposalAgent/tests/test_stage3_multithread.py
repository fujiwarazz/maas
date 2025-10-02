import sys
import os
import asyncio
import threading
import concurrent.futures
import time
from typing import Dict, Any, List
from dataclasses import dataclass

# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from proposalAgent.agents.utils.agent_states import AgentState
from proposalAgent.agents.utils.agent_utils import Toolkit
from proposalAgent.graphs.conditional_logic import ConditionalLogic
from proposalAgent.agents.stage3.completeness_checker import create_completeness_checker_agent
from proposalAgent.agents.stage3.feedback_analysis_agent import create_feedback_analysis_agent
from proposalAgent.agents.stage3.generator import create_generator_agent
from proposalAgent.agents.stage3.reflection_agent import create_reflection_agent
from proposalAgent.agents.stage3.final_analysis import create_final_analyst_agent
from proposalAgent.model_config import TONGYI_CONFIG
from proposalAgent.agents.utils.agent_utils import create_msg_delete

@dataclass
class TestResult:
    """测试结果数据类"""
    thread_id: str
    success: bool
    duration: float
    result_data: Dict[str, Any]
    error: str = None

class Stage3MultiThreadTest:
    """Stage3多线程测试类"""
    
    def __init__(self, max_workers: int = 3):
        """
        初始化多线程测试类
        
        Args:
            max_workers: 最大线程数
        """
        self.max_workers = max_workers
        self.config = TONGYI_CONFIG
        self.toolkit = Toolkit(config=self.config)
        self.deep_think_llm = ChatOpenAI(
            model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=TONGYI_CONFIG.get("api_key")
        )
        self.conditional_logic = ConditionalLogic()
        
        # 创建智能体
        self.final_analyst_node = create_final_analyst_agent(self.deep_think_llm)
        self.completeness_checker_node = create_completeness_checker_agent(self.deep_think_llm)
        self.feedback_analysis_node = create_feedback_analysis_agent(self.deep_think_llm)
        self.generator_node = create_generator_agent(self.deep_think_llm)
        self.reflection_node = create_reflection_agent(self.deep_think_llm)
        
        print(f"✅ 初始化多线程测试环境 (最大线程数: {max_workers})")
    
    def create_workflow(self, thread_id: str) -> StateGraph:
        """为每个线程创建独立的工作流"""
        workflow = StateGraph(AgentState)
        
        # 人类审核节点
        def human_review_node(state: AgentState) -> AgentState:
            print(f"[线程{thread_id}] 👤 等待人类审核...")
            
            # 显示检查内容
            print(f"[线程{thread_id}] 📋 人类审核 - 需要检查的内容:")
            print(f"[线程{thread_id}] 🔍 项目基本信息:")
            print(f"[线程{thread_id}]    研究主题: {state.get('research_topic', 'N/A')}")
            print(f"[线程{thread_id}]    申请人: {state.get('research_person_info', 'N/A')}")
            
            # 显示完备性检查结果
            completeness_result = state.get('completeness_check_result', {})
            print(f"[线程{thread_id}] 🔍 完备性检查结果:")
            print(f"[线程{thread_id}]    完整性: {completeness_result.get('is_complete', 'N/A')}")
            print(f"[线程{thread_id}]    质量评分: {completeness_result.get('overall_quality', 'N/A')}/5")
            
            # 模拟人类反馈
            if not state.get('human_feedback'):
                missing_parts = completeness_result.get('missing_parts', [])
                if "学术分析" in str(missing_parts):
                    state['human_feedback'] = f"[线程{thread_id}] 学术分析部分需要更深入"
                else:
                    state['human_feedback'] = f"[线程{thread_id}] 整体分析质量很好，可以直接生成报告"
            
            print(f"[线程{thread_id}] 📝 收到人类反馈: {state['human_feedback']}")
            return state
        
        # 路由函数
        def _route_after_human_review(state: AgentState) -> str:
            if state.get('skip_human_review'):
                print(f"[线程{thread_id}] 🔄 路由决策: 直接生成报告")
                return "generate"
            else:
                print(f"[线程{thread_id}] 🔄 路由决策: 分析人类反馈")
                return "feedback_analysis"
        
        def _route_after_feedback(state: AgentState) -> str:
            routing_decision = state.get('feedback_routing_decision', 'generate')
            print(f"[线程{thread_id}] 🔄 反馈分析路由决策: {routing_decision}")
            return routing_decision
        
        def _route_after_completeness(state: AgentState) -> str:
            recommendation = state.get('completeness_recommendation', 'need_human_review')
            if recommendation == 'complete':
                print(f"[线程{thread_id}] 🔄 完备性检查通过，直接生成报告")
                return "generate"
            else:
                print(f"[线程{thread_id}] 🔄 完备性检查未通过，需要人类审核")
                return "human_review"
        
        # 添加节点
        workflow.add_node("final_analyst_node", self.final_analyst_node)
        workflow.add_node("completeness_checker_node", self.completeness_checker_node)
        workflow.add_node("human_review_node", human_review_node)
        workflow.add_node("feedback_analysis_node", self.feedback_analysis_node)
        workflow.add_node("generator_node", self.generator_node)
        workflow.add_node("reflection_node", self.reflection_node)
        
        # 设置工作流路径
        workflow.add_edge(START, "final_analyst_node")
        workflow.add_edge("final_analyst_node", "completeness_checker_node")
        
        # 条件路由
        workflow.add_conditional_edges(
            "completeness_checker_node",
            _route_after_completeness,
            {
                "generate": "generator_node",
                "human_review": "human_review_node",
            },
        )
        
        workflow.add_conditional_edges(
            "human_review_node",
            _route_after_human_review,
            {
                "generate": "generator_node",
                "feedback_analysis": "feedback_analysis_node",
            },
        )
        
        workflow.add_conditional_edges(
            "feedback_analysis_node",
            _route_after_feedback,
            {
                "academic_analysis": "generator_node",
                "social_analysis": "generator_node",
                "future_influence": "generator_node",
                "interdisciplinary": "generator_node",
                "debate": "generator_node",
                "generate": "generator_node",
            },
        )
        
        workflow.add_edge("generator_node", "reflection_node")
        workflow.add_edge("reflection_node", END)
        
        # 编译工作流
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    
    def create_test_data(self, thread_id: str) -> Dict[str, Any]:
        """为每个线程创建测试数据"""
        return {
            "messages": [],
            "research_topic": f"基于大语言模型的智能教育系统研究 (线程{thread_id})",
            "research_basic_info": f"""
            申请人: 张三，清华大学计算机系副教授 (线程{thread_id})
            申请代码: F0212 数据科学与大数据计算
            研究周期: 3年
            申请金额: 80万元
            研究团队: 5人，包括2名博士生，2名硕士生，1名本科生
            """,
            "research_structure": "未提供",
            "research_person_info": f"张三，清华大学计算机系副教授 (线程{thread_id})",
            "research_project_team_info": "5人团队，包括2名博士生，2名硕士生，1名本科生",
            "research_project_apply_info": "申请代码: F0212，研究周期: 3年，申请金额: 80万元",
            "research_report_body_summary": "未提供",
            
            "academic_analysis_report": f"""
            学术能力评估 (线程{thread_id}):
            - 申请人具有扎实的计算机科学背景，在自然语言处理领域有丰富经验
            - 已发表SCI论文15篇，其中一区论文8篇
            - 主持过2项国家自然科学基金项目
            - 在Transformer架构优化方面有重要贡献
            """,
            "social_analysis_report": f"""
            社会影响分析 (线程{thread_id}):
            - 教育领域数字化转型的重要技术支撑
            - 有助于提升教育公平性和个性化学习
            - 可能对传统教育模式产生深远影响
            - 需要关注数据隐私和算法公平性问题
            """,
            "future_influence_report": f"""
            未来影响预测 (线程{thread_id}):
            - 短期(1-2年): 在教育辅助工具方面有应用前景
            - 中期(3-5年): 可能推动个性化教育模式普及
            - 长期(5-10年): 有望重塑教育生态系统
            - 风险: 技术依赖、数字鸿沟、伦理问题
            """,
            
            "interdisciplinary_results": ["教育学", "心理学", "伦理学", "数据科学"],
            "current_discipline": "计算机科学",
            "debate_results": {
                "feasibility": {
                    "judge_summary": f"技术基础扎实，团队配置合理，预期目标可实现 (线程{thread_id})",
                    "full_history": "可行性辩论：正方认为技术成熟，反方担心资源不足，裁判认为可行"
                },
                "innovation": {
                    "judge_summary": f"在个性化教育和大模型结合方面有创新点 (线程{thread_id})",
                    "full_history": "创新性辩论：正方强调技术融合创新，反方质疑创新程度，裁判认为有创新"
                }
            },
            
            "final_analysis_summary": f"""
            综合分析 (线程{thread_id})：
            该项目在技术可行性、团队配置、创新性方面表现良好，但在社会影响评估和风险分析方面需要进一步完善。
            建议加强跨学科协作，完善伦理审查机制。
            """,
            
            "completeness_check_result": {},
            "is_analysis_complete": None,
            "is_analysis_consistent": None,
            "completeness_recommendation": "",
            "skip_human_review": None,
            
            "reflection_decision": "",
            "human_feedback": "",
            
            "feedback_analysis_result": {},
            "feedback_routing_decision": "",
            "feedback_instructions": "",
            
            "final_report": ""
        }
    
    async def run_single_test(self, thread_id: str) -> TestResult:
        """运行单个测试"""
        start_time = time.time()
        
        try:
            print(f"[线程{thread_id}] 🚀 开始Stage3工作流测试")
            
            # 创建工作流
            graph = self.create_workflow(thread_id)
            
            # 创建测试数据
            state = self.create_test_data(thread_id)
            
            # 运行工作流
            config = {"configurable": {"thread_id": f"stage3_test_thread_{thread_id}"}}
            result = await graph.ainvoke(state, config=config)
            
            duration = time.time() - start_time
            
            print(f"[线程{thread_id}] ✅ 测试完成，耗时: {duration:.2f}秒")
            print(f"[线程{thread_id}] 📊 最终报告长度: {len(result.get('final_report', ''))}")
            
            return TestResult(
                thread_id=thread_id,
                success=True,
                duration=duration,
                result_data=result
            )
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"[线程{thread_id}] ❌ 测试失败: {e}")
            
            return TestResult(
                thread_id=thread_id,
                success=False,
                duration=duration,
                result_data={},
                error=str(e)
            )
    
    async def run_parallel_tests(self, num_tests: int = 3) -> List[TestResult]:
        """并行运行多个测试"""
        print(f"🚀 开始并行运行 {num_tests} 个Stage3工作流测试")
        print(f"📊 使用 {self.max_workers} 个线程")
        print("="*60)
        
        # 创建任务
        tasks = []
        for i in range(num_tests):
            thread_id = f"T{i+1}"
            task = self.run_single_test(thread_id)
            tasks.append(task)
        
        # 并行执行
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.time() - start_time
        
        # 处理结果
        test_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                test_results.append(TestResult(
                    thread_id=f"T{i+1}",
                    success=False,
                    duration=0,
                    result_data={},
                    error=str(result)
                ))
            else:
                test_results.append(result)
        
        # 显示结果
        print("\n" + "="*60)
        print("🎉 并行测试完成!")
        print("="*60)
        print(f"📊 总耗时: {total_duration:.2f}秒")
        print(f"📊 平均耗时: {total_duration/num_tests:.2f}秒/测试")
        
        success_count = sum(1 for r in test_results if r.success)
        print(f"📊 成功率: {success_count}/{num_tests} ({success_count/num_tests*100:.1f}%)")
        
        print(f"\n📋 详细结果:")
        for result in test_results:
            status = "✅ 成功" if result.success else "❌ 失败"
            print(f"   {result.thread_id}: {status} ({result.duration:.2f}秒)")
            if not result.success and result.error:
                print(f"      错误: {result.error}")
        
        return test_results
    
    def run_threading_tests(self, num_tests: int = 3) -> List[TestResult]:
        """使用threading模块运行测试"""
        print(f"🚀 开始使用threading运行 {num_tests} 个Stage3工作流测试")
        print(f"📊 使用 {self.max_workers} 个线程")
        print("="*60)
        
        results = []
        threads = []
        
        def run_test_thread(thread_id: str):
            """线程函数"""
            try:
                # 在新线程中运行异步任务
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.run_single_test(thread_id))
                results.append(result)
                loop.close()
            except Exception as e:
                results.append(TestResult(
                    thread_id=thread_id,
                    success=False,
                    duration=0,
                    result_data={},
                    error=str(e)
                ))
        
        # 创建并启动线程
        start_time = time.time()
        for i in range(num_tests):
            thread_id = f"T{i+1}"
            thread = threading.Thread(target=run_test_thread, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        total_duration = time.time() - start_time
        
        # 显示结果
        print("\n" + "="*60)
        print("🎉 Threading测试完成!")
        print("="*60)
        print(f"📊 总耗时: {total_duration:.2f}秒")
        print(f"📊 平均耗时: {total_duration/num_tests:.2f}秒/测试")
        
        success_count = sum(1 for r in results if r.success)
        print(f"📊 成功率: {success_count}/{num_tests} ({success_count/num_tests*100:.1f}%)")
        
        print(f"\n📋 详细结果:")
        for result in results:
            status = "✅ 成功" if result.success else "❌ 失败"
            print(f"   {result.thread_id}: {status} ({result.duration:.2f}秒)")
            if not result.success and result.error:
                print(f"      错误: {result.error}")
        
        return results

async def main():
    """主函数"""
    # 创建多线程测试实例
    test = Stage3MultiThreadTest(max_workers=3)
    
    print("选择测试模式:")
    print("1. 异步并行测试 (推荐)")
    print("2. Threading测试")
    print("3. 两种都运行")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 异步并行测试
        await test.run_parallel_tests(num_tests=3)
    elif choice == "2":
        # Threading测试
        test.run_threading_tests(num_tests=3)
    elif choice == "3":
        # 两种都运行
        print("\n🔄 运行异步并行测试...")
        await test.run_parallel_tests(num_tests=3)
        
        print("\n🔄 运行Threading测试...")
        test.run_threading_tests(num_tests=3)
    else:
        print("无效选择，运行异步并行测试...")
        await test.run_parallel_tests(num_tests=3)

if __name__ == "__main__":
    asyncio.run(main())
