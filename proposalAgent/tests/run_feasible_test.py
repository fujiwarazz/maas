#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可行性辩论测试运行脚本
"""

import sys
import os

from .test_feasible_debate_graph import FeasibleDebateTestGraph


def main():
    """运行可行性辩论测试"""
    print("🚀 启动可行性辩论Agent测试")
    print("=" * 60)
    
    # 创建测试实例
    test_runner = FeasibleDebateTestGraph()
    
    try:
        # 运行测试
        print("开始执行测试...")
        test_results = test_runner.run_comprehensive_test()
        
        # 检查结果
        all_passed = all(success for _, success in test_results)
        
        if all_passed:
            print("\n🎉 所有测试通过！")
            return 0
        else:
            print("\n❌ 部分测试失败")
            return 1
            
    except Exception as e:
        print(f"\n💥 测试运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
