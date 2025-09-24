#!/usr/bin/env python3
"""
测试简化工作流
"""

from simple_proposal_runner import create_simple_runner
from proposalAgent.utils.logger import get_logger

logger = get_logger("TestSimpleWorkflow")


def test_simple_workflow():
    """测试简化工作流是否能正常运行"""
    print("=== 测试简化工作流 ===")
    
    try:
        # 创建运行器
        print("1. 创建运行器...")
        runner = create_simple_runner()
        print("✓ 运行器创建成功")
        
        # 获取工作流信息
        print("2. 获取工作流信息...")
        info = runner.get_info()
        print(f"✓ 工作流信息: {info}")
        
        # 测试简单的graph.invoke()调用
        print("3. 测试graph.invoke()...")
        
        # 创建测试输入 - 明确表示这是项目评估请求
        test_input = {
            "messages": [("user", "请对人工智能在医疗领域的项目申请书进行全面评估，包括可行性分析、创新性评价、学术价值评估和社会影响分析")],
            "research_topic": ["人工智能", "医疗"],
            "filepath": ""
            # 不设置intention_decision，让系统自动判断
        }
        
        # 调用图 - 提供必要的配置
        config = {"configurable": {"thread_id": "test_thread_123"}}
        result = runner.graph.invoke(test_input, config=config)
        
        print("✓ graph.invoke()调用成功")
        print(f"结果类型: {type(result)}")
        print(f"结果键: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
        
        # 检查关键结果
        if isinstance(result, dict):
            final_report = result.get("final_report", "")
            if final_report:
                print(f"✓ 生成了最终报告 ({len(final_report)} 字符)")
                print(f"报告预览: {final_report[:200]}...")
            else:
                print("⚠ 未生成最终报告")
            
            # 检查其他分析结果
            academic_report = result.get("academic_analysis_report", "")
            social_report = result.get("social_analysis_report", "")
            future_report = result.get("future_influence_report", "")
            
            print(f"学术分析报告: {'✓' if academic_report else '✗'}")
            print(f"社会分析报告: {'✓' if social_report else '✗'}")
            print(f"未来影响报告: {'✓' if future_report else '✗'}")
        
        print("\n=== 测试成功！简化工作流可以正常运行 ===")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        logger.error("测试过程中发生错误: %s", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_convenience_method():
    """测试便捷方法"""
    print("\n=== 测试便捷方法 ===")
    
    try:
        runner = create_simple_runner()
        
        # 使用便捷方法
        result = runner.evaluate(
            user_prompt="请评估区块链技术在金融行业的可行性",
            user_interests=["技术可行性", "市场前景"]
        )
        
        print(f"✓ 便捷方法调用成功")
        print(f"状态: {result.get('status')}")
        print(f"线程ID: {result.get('thread_id')}")
        
        if result.get('status') == 'completed':
            print("✓ 评估完成")
        else:
            print(f"⚠ 评估状态: {result.get('status')}")
            if result.get('error'):
                print(f"错误: {result.get('error')}")
        
        return True
        
    except Exception as e:
        print(f"✗ 便捷方法测试失败: {e}")
        return False


if __name__ == "__main__":
    success1 = test_simple_workflow()
    success2 = test_convenience_method()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！简化工作流已准备就绪。")
        print("\n使用方法:")
        print("1. 直接使用: runner = create_simple_runner(); result = runner.graph.invoke(input)")
        print("2. 便捷方法: runner.evaluate('your prompt')")
    else:
        print("\n❌ 部分测试失败，请检查配置和依赖。")
