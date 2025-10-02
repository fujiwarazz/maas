import sys
import os
import asyncio
import time
import threading
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langchain_openai import ChatOpenAI
from proposalAgent.agents.utils.agent_states import AgentState
from proposalAgent.agents.stage3.completeness_checker import create_completeness_checker_agent
from proposalAgent.agents.stage3.feedback_analysis_agent import create_feedback_analysis_agent
from proposalAgent.agents.stage3.generator import create_generator_agent
from proposalAgent.agents.stage3.reflection_agent import create_reflection_agent
from proposalAgent.agents.stage3.final_analysis import create_final_analyst_agent
from proposalAgent.model_config import TONGYI_CONFIG

class SimpleMultiThreadTest:
    """简化的多线程Stage3测试类"""
    
    def __init__(self, max_workers: int = 2):
        """
        初始化多线程测试类
        
        Args:
            max_workers: 最大线程数，建议不超过2以避免API限制
        """
        self.max_workers = max_workers
        self.config = TONGYI_CONFIG
        
        # 为每个线程创建独立的LLM实例
        self.llm_instances = []
        for i in range(max_workers):
            llm = ChatOpenAI(
                model="qwen-plus",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=TONGYI_CONFIG.get("api_key"),
                request_timeout=60,  # 增加超时时间
                max_retries=2  # 增加重试次数
            )
            self.llm_instances.append(llm)
        
        print(f"✅ 初始化多线程测试环境 (最大线程数: {max_workers})")
    
    def create_agents(self, thread_id: int):
        """为每个线程创建独立的智能体实例"""
        llm = self.llm_instances[thread_id % len(self.llm_instances)]
        
        return {
            'final_analyst': create_final_analyst_agent(llm),
            'completeness_checker': create_completeness_checker_agent(llm),
            'feedback_analysis': create_feedback_analysis_agent(llm),
            'generator': create_generator_agent(llm),
            'reflection': create_reflection_agent(llm)
        }
    
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
    
    def run_single_test_sync(self, thread_id: str) -> Dict[str, Any]:
        """同步运行单个测试"""
        start_time = time.time()
        
        try:
            print(f"[{thread_id}] 🚀 开始测试")
            
            # 创建独立的智能体实例
            agents = self.create_agents(int(thread_id.replace('T', '')) - 1)
            
            # 创建测试数据
            state = self.create_test_data(thread_id)
            
            # 模拟工作流执行
            print(f"[{thread_id}] 📊 执行最终分析...")
            state = agents['final_analyst'](state)
            
            print(f"[{thread_id}] 🔍 执行完备性检查...")
            state = agents['completeness_checker'](state)
            
            # 模拟人类审核
            print(f"[{thread_id}] 👤 人类审核节点")
            completeness_result = state.get('completeness_check_result', {})
            print(f"[{thread_id}] 📋 项目: {state.get('research_topic', 'N/A')}")
            print(f"[{thread_id}] 🔍 完备性: {completeness_result.get('is_complete', 'N/A')}")
            
            # 模拟人类反馈
            state['human_feedback'] = f"[{thread_id}] 分析质量良好，可以生成报告"
            print(f"[{thread_id}] 📝 反馈: {state['human_feedback']}")
            
            print(f"[{thread_id}] 🔄 执行反馈分析...")
            state = agents['feedback_analysis'](state)
            
            print(f"[{thread_id}] 📝 生成最终报告...")
            state = agents['generator'](state)
            
            print(f"[{thread_id}] 🤔 执行反思...")
            state = agents['reflection'](state)
            
            duration = time.time() - start_time
            report_length = len(state.get('final_report', ''))
            
            print(f"[{thread_id}] ✅ 测试完成，耗时: {duration:.2f}秒，报告长度: {report_length}")
            
            return {
                "thread_id": thread_id,
                "success": True,
                "duration": duration,
                "report_length": report_length,
                "result": state
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
    
    def run_thread_pool_tests(self, num_tests: int = 3) -> List[Dict[str, Any]]:
        """使用线程池运行测试"""
        print(f"🚀 开始使用线程池运行 {num_tests} 个Stage3工作流测试")
        print(f"📊 最大线程数: {self.max_workers}")
        print("="*60)
        
        # 使用线程池执行
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_thread = {
                executor.submit(self.run_single_test_sync, f"T{i+1}"): f"T{i+1}" 
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
    
    def run_sequential_tests(self, num_tests: int = 3) -> List[Dict[str, Any]]:
        """顺序运行测试（作为对比）"""
        print(f"🚀 开始顺序运行 {num_tests} 个Stage3工作流测试")
        print("="*60)
        
        start_time = time.time()
        results = []
        
        for i in range(num_tests):
            thread_id = f"T{i+1}"
            result = self.run_single_test_sync(thread_id)
            results.append(result)
            
            # 在测试之间添加短暂延迟
            if i < num_tests - 1:
                print(f"⏳ 等待2秒后开始下一个测试...")
                time.sleep(2)
        
        total_duration = time.time() - start_time
        
        # 显示结果
        print("\n" + "="*60)
        print("🎉 顺序测试完成!")
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

def main():
    """主函数"""
    print("选择测试模式:")
    print("1. 顺序测试 (稳定)")
    print("2. 多线程测试 (2个线程)")
    print("3. 两种都运行 (对比)")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 顺序测试
        test = SimpleMultiThreadTest(max_workers=1)
        test.run_sequential_tests(num_tests=3)
    elif choice == "2":
        # 多线程测试
        test = SimpleMultiThreadTest(max_workers=2)
        test.run_thread_pool_tests(num_tests=3)
    elif choice == "3":
        # 两种都运行
        print("\n🔄 运行顺序测试...")
        test1 = SimpleMultiThreadTest(max_workers=1)
        test1.run_sequential_tests(num_tests=3)
        
        print("\n🔄 运行多线程测试...")
        test2 = SimpleMultiThreadTest(max_workers=2)
        test2.run_thread_pool_tests(num_tests=3)
    else:
        print("无效选择，运行顺序测试...")
        test = SimpleMultiThreadTest(max_workers=1)
        test.run_sequential_tests(num_tests=3)

if __name__ == "__main__":
    main()
