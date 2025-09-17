#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage1 图结构测试脚本
"""

import sys
import os
import pathlib
import json

# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from proposalAgent.graphs.stage1_graph import create_stage1_graph, create_streaming_stage1_graph


def test_intention_recognition():
    """测试意图识别功能"""
    print("🧠 测试意图识别功能")
    print("=" * 50)
    
    graph = create_streaming_stage1_graph()
    
    test_cases = [
        {
            "name": "通用对话测试",
            "input": "你好，今天天气怎么样？",
            "expected": "output"
        },
        {
            "name": "申请书分析测试",
            "input": "请分析这篇科研申请书的结构和内容",
            "expected": "structure"
        },
        {
            "name": "论文评估测试", 
            "input": "评估一下这篇论文的创新性和可行性",
            "expected": "structure"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}: {case['name']}")
        print(f"输入: {case['input']}")
        print("-" * 30)
        
        state = {
            "messages": [("user", case['input'])],
            "intention_decision": "",
            "file_path": "",
            "research_structure": "",
            "research_person_info": "",
            "research_project_team_info": "",
            "research_project_apply_info": "",
            "research_report_body_summary": ""
        }
        
        try:
            result = graph.invoke(state)
            intention = result.get("intention_decision", "").lower()
            print(f"识别意图: {intention}")
            
            if case['expected'] in intention:
                print("✅ 意图识别正确")
            else:
                print("❌ 意图识别错误")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")


def test_output_generation():
    """测试输出生成功能"""
    print("\n💬 测试输出生成功能")
    print("=" * 50)
    
    graph = create_streaming_stage1_graph()
    
    test_cases = [
        "你好，请介绍一下自己",
        "什么是人工智能？",
        "请解释一下机器学习的基本概念"
    ]
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}: {question}")
        print("-" * 30)
        
        state = {
            "messages": [("user", question)],
            "intention_decision": "",
            "file_path": "",
            "research_structure": "",
            "research_person_info": "",
            "research_project_team_info": "",
            "research_project_apply_info": "",
            "research_report_body_summary": ""
        }
        
        try:
            result = graph.invoke(state)
            messages = result.get("messages", [])
            
            
            if messages:
                response = messages[-1]
                if isinstance(response, str):
                    print(f"回复长度: {len(response)} 字符")
                    print(f"回复预览: {response[:100]}...")
                    print("✅ 输出生成成功")
                else:
                    print(f"回复类型: {type(response)}")
                    print("✅ 输出生成成功")
            else:
                print("❌ 没有生成输出")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")


def test_structure_analysis():
    """测试结构分析功能"""
    print("\n📄 测试结构分析功能")
    print("=" * 50)
    
    # 查找可用的PDF文件
    possible_paths = [
        "/Users/peelsannaw/Desktop/提交版本.pdf",
        "/Users/peelsannaw/Desktop/codes/maas/mas4proposal/test.pdf",
        "./test.pdf"
    ]
    
    pdf_path = None
    for path in possible_paths:
        if pathlib.Path(path).exists():
            pdf_path = path
            break
    
    if not pdf_path:
        print("⚠️  未找到测试用PDF文件，跳过结构分析测试")
        print("可尝试的路径:")
        for path in possible_paths:
            print(f"  - {path}")
        return
    
    print(f"使用PDF文件: {pdf_path}")
    
    graph = create_streaming_stage1_graph()
    
    state = {
        "messages": [("user", "分析这篇申请书的结构和内容")],
        "intention_decision": "",
        "file_path": pdf_path,
        "research_structure": "",
        "research_person_info": "",
        "research_project_team_info": "",
        "research_project_apply_info": "",
        "research_report_body_summary": ""
    }
    
    try:
        print("开始结构分析...")
        result = graph.invoke(state)
        
        # 检查输出字段
        fields = [
            "research_structure",
            "research_person_info", 
            "research_project_team_info",
            "research_project_apply_info",
            "research_report_body_summary"
        ]
        
        print("\n结构分析结果:")
        for field in fields:
            content = result.get(field, "")
            if content:
                print(f"✅ {field}: {len(content)} 字符")
                # 显示前100个字符的预览
                preview = content[:100].replace('\n', ' ')
                print(f"   预览: {preview}...")
            else:
                print(f"❌ {field}: 无内容")
        
        print("✅ 结构分析测试完成")
        
    except Exception as e:
        print(f"❌ 结构分析测试失败: {e}")


def test_graph_flow():
    """测试完整的图流程"""
    print("\n🔄 测试完整图流程")
    print("=" * 50)
    
    graph = create_streaming_stage1_graph()
    
    # 测试不同的输入场景
    scenarios = [
        {
            "name": "场景1: 普通问答",
            "messages": [("user", "请解释什么是深度学习")],
            "file_path": ""
        },
        {
            "name": "场景2: 申请书相关询问", 
            "messages": [("user", "如何写好一份科研申请书？")],
            "file_path": ""
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print("-" * 30)
        
        state = {
            "messages": scenario["messages"],
            "intention_decision": "",
            "file_path": scenario["file_path"],
            "research_structure": "",
            "research_person_info": "",
            "research_project_team_info": "",
            "research_project_apply_info": "",
            "research_report_body_summary": ""
        }
        
        try:
            result = graph.invoke(state)
            
            print("执行路径分析:")
            print(f"- 意图识别: {result.get('intention_decision', 'N/A')}")
            
            if result.get("messages"):
                print(f"- 最终输出: 成功 ({len(str(result['messages'][-1]))} 字符)")
            
            if result.get("research_structure"):
                print(f"- 结构分析: 成功 ({len(result['research_structure'])} 字符)")
            
            print("✅ 流程测试完成")
            
        except Exception as e:
            print(f"❌ 流程测试失败: {e}")


def main():
    """主测试函数"""
    print("🚀 开始 Stage1 图结构测试")
    print("=" * 60)
    
    try:
        # test_intention_recognition()
        test_output_generation()
        # test_structure_analysis()
        # test_graph_flow()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")


if __name__ == "__main__":
    main()
