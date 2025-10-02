import sys
import os
import asyncio
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from proposalAgent.agents.utils.agent_states import AgentState
from proposalAgent.agents.stage3.completeness_checker import create_completeness_checker_agent
from proposalAgent.agents.stage3.feedback_analysis_agent import create_feedback_analysis_agent
from proposalAgent.agents.stage3.generator import create_generator_agent
from proposalAgent.agents.stage3.reflection_agent import create_reflection_agent
from proposalAgent.agents.stage3.final_analysis import create_final_analyst_agent
from proposalAgent.model_config import TONGYI_CONFIG

class ConcurrentStage3Test:
    """并发Stage3测试类"""
    
    def __init__(self):
        self.config = TONGYI_CONFIG
        self.deep_think_llm = ChatOpenAI(
            model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=TONGYI_CONFIG.get("api_key")
        )
        
        # 创建智能体
        self.final_analyst_node = create_final_analyst_agent(self.deep_think_llm)
        self.completeness_checker_node = create_completeness_checker_agent(self.deep_think_llm)
        self.feedback_analysis_node = create_feedback_analysis_agent(self.deep_think_llm)
        self.generator_node = create_generator_agent(self.deep_think_llm)
        self.reflection_node = create_reflection_agent(self.deep_think_llm)
        
        print("✅ 初始化并发测试环境")
    
    def create_workflow(self) -> StateGraph:
        """创建工作流"""
        workflow = StateGraph(AgentState)
        
        # 人类审核节点
        def human_review_node(state: AgentState) -> AgentState:
            thread_id = state.get('thread_id', 'Unknown')
            print(f"[{thread_id}] 👤 人类审核节点")
            
            # 显示检查内容
            print(f"[{thread_id}] 📋 项目: {state.get('research_topic', 'N/A')}")
            completeness_result = state.get('completeness_check_result', {})
            print(f"[{thread_id}] 🔍 完备性: {completeness_result.get('is_complete', 'N/A')}")
            
            # 模拟人类反馈
            if not state.get('human_feedback'):
                state['human_feedback'] = f"[{thread_id}] 分析质量良好，可以生成报告"
            
            print(f"[{thread_id}] 📝 反馈: {state['human_feedback']}")
            return state
        
        # 路由函数
        def _route_after_human_review(state: AgentState) -> str:
            return "feedback_analysis"
        
        def _route_after_feedback(state: AgentState) -> str:
            return "generate"
        
        def _route_after_completeness(state: AgentState) -> str:
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
        
        workflow.add_conditional_edges(
            "completeness_checker_node",
            _route_after_completeness,
            {"human_review": "human_review_node"}
        )
        
        workflow.add_conditional_edges(
            "human_review_node",
            _route_after_human_review,
            {"feedback_analysis": "feedback_analysis_node"}
        )
        
        workflow.add_conditional_edges(
            "feedback_analysis_node",
            _route_after_feedback,
            {"generate": "generator_node"}
        )
        
        workflow.add_edge("generator_node", "reflection_node")
        workflow.add_edge("reflection_node", END)
        
        # 编译工作流
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    
    def create_test_data(self, thread_id: str) -> Dict[str, Any]:
        """创建测试数据"""
        return {
            "thread_id": thread_id,
            "messages": [],
            "research_topic": f"基于大语言模型的智能教育系统研究 (线程{thread_id})",
            "research_person_info": f"张三，清华大学计算机系副教授 (线程{thread_id})",
            "research_project_team_info": "5人团队，包括2名博士生，2名硕士生，1名本科生",
            "research_project_apply_info": "申请代码: F0212，研究周期: 3年，申请金额: 80万元",
            
            "academic_analysis_report": f"""
            学术能力评估 (线程{thread_id}):
            - 申请人具有扎实的计算机科学背景
            - 已发表SCI论文15篇，其中一区论文8篇
            - 主持过2项国家自然科学基金项目
            """,
            "social_analysis_report": f"""
            社会影响分析 (线程{thread_id}):
            - 教育领域数字化转型的重要技术支撑
            - 有助于提升教育公平性和个性化学习
            """,
            "future_influence_report": f"""
            未来影响预测 (线程{thread_id}):
            - 短期: 在教育辅助工具方面有应用前景
            - 中期: 可能推动个性化教育模式普及
            """,
            
            "interdisciplinary_results": ["教育学", "心理学", "伦理学", "数据科学"],
            "debate_results": {
                "feasibility": {"judge_summary": f"技术基础扎实，预期目标可实现 (线程{thread_id})"},
                "innovation": {"judge_summary": f"在个性化教育和大模型结合方面有创新点 (线程{thread_id})"}
            },
            
            "final_analysis_summary": f"""
            综合分析 (线程{thread_id})：
            该项目在技术可行性、团队配置、创新性方面表现良好。
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
    
    async def run_single_test(self, thread_id: str) -> Dict[str, Any]:
        """运行单个测试"""
        start_time = time.time()
        
        try:
            print(f"[{thread_id}] 🚀 开始测试")
            
            # 创建工作流
            graph = self.create_workflow()
            
            # 创建测试数据
            state = self.create_test_data(thread_id)
            
            # 运行工作流
            config = {"configurable": {"thread_id": f"stage3_test_{thread_id}"}}
            result = await graph.ainvoke(state, config=config)
            
            duration = time.time() - start_time
            report_length = len(result.get('final_report', ''))
            
            print(f"[{thread_id}] ✅ 测试完成，耗时: {duration:.2f}秒，报告长度: {report_length}")
            
            return {
                "thread_id": thread_id,
                "success": True,
                "duration": duration,
                "report_length": report_length,
                "result": result
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"[{thread_id}] ❌ 测试失败: {e}")
            
            return {
                "thread_id": thread_id,
                "success": False,
                "duration": duration,
                "error": str(e)
            }
    
    async def run_concurrent_tests(self, num_tests: int = 5) -> List[Dict[str, Any]]:
        """并发运行多个测试"""
        print(f"🚀 开始并发运行 {num_tests} 个Stage3工作流测试")
        print("="*60)
        
        # 创建任务
        tasks = []
        for i in range(num_tests):
            thread_id = f"T{i+1}"
            task = self.run_single_test(thread_id)
            tasks.append(task)
        
        # 并发执行
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.time() - start_time
        
        # 处理结果
        test_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                test_results.append({
                    "thread_id": f"T{i+1}",
                    "success": False,
                    "duration": 0,
                    "error": str(result)
                })
            else:
                test_results.append(result)
        
        # 显示结果
        print("\n" + "="*60)
        print("🎉 并发测试完成!")
        print("="*60)
        print(f"📊 总耗时: {total_duration:.2f}秒")
        print(f"📊 平均耗时: {total_duration/num_tests:.2f}秒/测试")
        
        success_count = sum(1 for r in test_results if r.get('success', False))
        print(f"📊 成功率: {success_count}/{num_tests} ({success_count/num_tests*100:.1f}%)")
        
        print(f"\n📋 详细结果:")
        for result in test_results:
            thread_id = result.get('thread_id', 'Unknown')
            success = result.get('success', False)
            duration = result.get('duration', 0)
            status = "✅ 成功" if success else "❌ 失败"
            
            if success:
                report_length = result.get('report_length', 0)
                print(f"   {thread_id}: {status} ({duration:.2f}秒, 报告{report_length}字符)")
            else:
                error = result.get('error', 'Unknown error')
                print(f"   {thread_id}: {status} ({duration:.2f}秒) - {error}")
        
        return test_results
    
    def run_thread_pool_tests(self, num_tests: int = 5, max_workers: int = 3) -> List[Dict[str, Any]]:
        """使用线程池运行测试"""
        print(f"🚀 开始使用线程池运行 {num_tests} 个Stage3工作流测试")
        print(f"📊 最大线程数: {max_workers}")
        print("="*60)
        
        def run_test_sync(thread_id: str) -> Dict[str, Any]:
            """同步运行测试"""
            start_time = time.time()
            
            try:
                print(f"[{thread_id}] 🚀 开始测试")
                
                # 在新线程中运行异步任务
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # 创建工作流
                graph = self.create_workflow()
                
                # 创建测试数据
                state = self.create_test_data(thread_id)
                
                # 运行工作流
                config = {"configurable": {"thread_id": f"stage3_test_{thread_id}"}}
                result = loop.run_until_complete(graph.ainvoke(state, config=config))
                
                loop.close()
                
                duration = time.time() - start_time
                report_length = len(result.get('final_report', ''))
                
                print(f"[{thread_id}] ✅ 测试完成，耗时: {duration:.2f}秒，报告长度: {report_length}")
                
                return {
                    "thread_id": thread_id,
                    "success": True,
                    "duration": duration,
                    "report_length": report_length,
                    "result": result
                }
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"[{thread_id}] ❌ 测试失败: {e}")
                
                return {
                    "thread_id": thread_id,
                    "success": False,
                    "duration": duration,
                    "error": str(e)
                }
        
        # 使用线程池执行
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_thread = {
                executor.submit(run_test_sync, f"T{i+1}"): f"T{i+1}" 
                for i in range(num_tests)
            }
            
            # 收集结果
            for future in as_completed(future_to_thread):
                thread_id = future_to_thread[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "thread_id": thread_id,
                        "success": False,
                        "duration": 0,
                        "error": str(e)
                    })
        
        total_duration = time.time() - start_time
        
        # 显示结果
        print("\n" + "="*60)
        print("🎉 线程池测试完成!")
        print("="*60)
        print(f"📊 总耗时: {total_duration:.2f}秒")
        print(f"📊 平均耗时: {total_duration/num_tests:.2f}秒/测试")
        
        success_count = sum(1 for r in results if r.get('success', False))
        print(f"📊 成功率: {success_count}/{num_tests} ({success_count/num_tests*100:.1f}%)")
        
        print(f"\n📋 详细结果:")
        for result in results:
            thread_id = result.get('thread_id', 'Unknown')
            success = result.get('success', False)
            duration = result.get('duration', 0)
            status = "✅ 成功" if success else "❌ 失败"
            
            if success:
                report_length = result.get('report_length', 0)
                print(f"   {thread_id}: {status} ({duration:.2f}秒, 报告{report_length}字符)")
            else:
                error = result.get('error', 'Unknown error')
                print(f"   {thread_id}: {status} ({duration:.2f}秒) - {error}")
        
        return results

async def main():
    """主函数"""
    test = ConcurrentStage3Test()
    
    print("选择测试模式:")
    print("1. 异步并发测试 (推荐)")
    print("2. 线程池测试")
    print("3. 两种都运行")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 异步并发测试
        await test.run_concurrent_tests(num_tests=5)
    elif choice == "2":
        # 线程池测试
        test.run_thread_pool_tests(num_tests=5, max_workers=3)
    elif choice == "3":
        # 两种都运行
        print("\n🔄 运行异步并发测试...")
        await test.run_concurrent_tests(num_tests=5)
        
        print("\n🔄 运行线程池测试...")
        test.run_thread_pool_tests(num_tests=5, max_workers=3)
    else:
        print("无效选择，运行异步并发测试...")
        await test.run_concurrent_tests(num_tests=5)

if __name__ == "__main__":
    asyncio.run(main())
