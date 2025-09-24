#!/usr/bin/env python3
"""
简化工作流使用示例
展示如何使用SimpleWorkflow进行项目评估
"""

from simple_proposal_runner import create_simple_runner
from proposalAgent.utils.logger import get_logger

logger = get_logger("ExampleUsage")


def example_1_basic_usage():
    """示例1：基本用法 - 使用graph.invoke()"""
    print("=== 示例1：基本用法 ===")
    
    # 创建运行器
    runner = create_simple_runner()
    
    # 准备输入
    input_data = {
        "messages": [("user", "请对区块链技术在供应链管理中的应用进行项目评估分析")],
        "research_topic": ["区块链", "供应链管理"],
        "filepath": ""
    }
    
    # 配置
    config = {"configurable": {"thread_id": "example_1_thread"}}
    
    try:
        # 调用图
        result = runner.graph.invoke(input_data, config=config)
        
        print("✓ 评估完成")
        print(f"最终报告长度: {len(result.get('final_report', ''))}")
        print(f"学术分析: {'✓' if result.get('academic_analysis_report') else '✗'}")
        print(f"社会分析: {'✓' if result.get('social_analysis_report') else '✗'}")
        print(f"未来影响: {'✓' if result.get('future_influence_report') else '✗'}")
        
        return result
        
    except Exception as e:
        print(f"✗ 评估失败: {e}")
        return None


def example_2_convenience_method():
    """示例2：便捷方法"""
    print("\n=== 示例2：便捷方法 ===")
    
    # 创建运行器
    runner = create_simple_runner()
    
    try:
        # 使用便捷方法
        result = runner.evaluate(
            user_prompt="请评估人工智能在教育领域的应用项目，重点关注技术可行性和教育效果",
            user_interests=["技术可行性", "教育效果", "实施成本"],
            filepath=""
        )
        
        print(f"✓ 评估状态: {result.get('status')}")
        
        if result.get('status') == 'completed':
            print("✓ 评估成功完成")
            academic = result.get('academic_analysis', '')
            if academic:
                print(f"学术分析摘要: {academic[:150]}...")
        else:
            print(f"⚠ 评估状态: {result.get('status')}")
            if result.get('error'):
                print(f"错误信息: {result.get('error')}")
        
        return result
        
    except Exception as e:
        print(f"✗ 便捷方法失败: {e}")
        return None


def example_3_with_file():
    """示例3：包含文件的评估（如果有PDF文件）"""
    print("\n=== 示例3：文件评估 ===")
    
    # 检查是否有示例文件
    import os
    example_file = "/Users/peelsannaw/Desktop/codes/maas/mas4proposal/data/提交版本.pdf"
    
    if os.path.exists(example_file):
        print(f"发现示例文件: {example_file}")
        
        runner = create_simple_runner()
        
        try:
            result = runner.evaluate(
                user_prompt="请对这份项目申请书进行全面评估分析",
                filepath=example_file,
                user_interests=["学术价值", "创新性", "可行性"]
            )
            
            print(f"✓ 文件评估状态: {result.get('status')}")
            return result
            
        except Exception as e:
            print(f"✗ 文件评估失败: {e}")
            return None
    else:
        print("⚠ 未找到示例PDF文件，跳过文件评估示例")
        return None


def example_4_step_by_step():
    """示例4：分步骤的详细使用"""
    print("\n=== 示例4：分步骤使用 ===")
    
    # 步骤1：创建运行器
    print("步骤1：创建运行器...")
    runner = create_simple_runner()
    print("✓ 运行器创建完成")
    
    # 步骤2：查看工作流信息
    print("步骤2：获取工作流信息...")
    info = runner.get_info()
    print(f"✓ LLM提供商: {info.get('llm_provider')}")
    print(f"✓ 深度思考模型: {info.get('deep_think_model')}")
    print(f"✓ 快速思考模型: {info.get('quick_think_model')}")
    
    # 步骤3：准备评估数据
    print("步骤3：准备评估数据...")
    evaluation_data = {
        "messages": [("user", "请评估智能制造系统在工业4.0中的应用项目")],
        "research_topic": ["智能制造", "工业4.0"],
        "filepath": ""
    }
    config = {"configurable": {"thread_id": "step_by_step_example"}}
    print("✓ 数据准备完成")
    
    # 步骤4：执行评估
    print("步骤4：执行评估...")
    try:
        result = runner.graph.invoke(evaluation_data, config=config)
        print("✓ 评估执行完成")
        
        # 步骤5：分析结果
        print("步骤5：分析结果...")
        if isinstance(result, dict):
            keys = list(result.keys())
            print(f"✓ 结果包含 {len(keys)} 个字段")
            
            # 检查关键结果
            key_fields = ['final_report', 'academic_analysis_report', 'social_analysis_report']
            for field in key_fields:
                value = result.get(field, '')
                status = "✓" if value else "✗"
                print(f"{status} {field}: {len(value)} 字符")
        
        return result
        
    except Exception as e:
        print(f"✗ 评估执行失败: {e}")
        return None


def main():
    """主函数：运行所有示例"""
    print("🚀 简化工作流使用示例")
    print("=" * 50)
    
    # 运行各个示例
    examples = [
        example_1_basic_usage,
        example_2_convenience_method,
        example_3_with_file,
        example_4_step_by_step
    ]
    
    results = []
    for example_func in examples:
        try:
            result = example_func()
            results.append(result)
        except Exception as e:
            print(f"示例 {example_func.__name__} 执行失败: {e}")
            results.append(None)
    
    print("\n" + "=" * 50)
    print("📊 示例执行总结:")
    success_count = sum(1 for r in results if r is not None)
    print(f"✓ 成功: {success_count}/{len(examples)}")
    print(f"✗ 失败: {len(examples) - success_count}/{len(examples)}")
    
    if success_count > 0:
        print("\n🎉 简化工作流运行正常！")
        print("\n📖 使用方法总结:")
        print("1. 基本用法: runner.graph.invoke(input_data, config)")
        print("2. 便捷方法: runner.evaluate(prompt, interests, filepath)")
        print("3. 获取信息: runner.get_info()")
    else:
        print("\n❌ 请检查配置和依赖")


if __name__ == "__main__":
    main()
