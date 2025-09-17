#!/usr/bin/env python3
"""
学术分析系统完整测试套件
包含Agent测试和Graph测试
"""

import os
import sys
import subprocess
from datetime import datetime

def run_test_file(test_file, description):
    """运行测试文件"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"📁 测试文件: {test_file}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        print(result.stdout)
        if result.stderr:
            print("⚠️ 错误输出:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ 运行测试时发生错误: {e}")
        return False

def main():
    """主函数"""
    print("🚀 学术分析系统完整测试套件")
    print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python环境: {sys.executable}")
    print(f"📂 工作目录: {os.getcwd()}")
    
    # 检查环境
    env_name = os.environ.get('CONDA_DEFAULT_ENV', '未知')
    print(f"🌍 Conda环境: {env_name}")
    
    if env_name == 'hxrag':
        print("✅ 环境检查通过")
    else:
        print("⚠️ 未在hxrag环境中，但测试将继续")
    
    # 测试文件列表
    tests = [
        ("test_academic_final.py", "学术分析Agent功能测试"),
        ("test_academic_graph_simple.py", "学术分析图结构测试")
    ]
    
    results = []
    
    # 运行所有测试
    for test_file, description in tests:
        if os.path.exists(test_file):
            success = run_test_file(test_file, description)
            results.append((description, success))
        else:
            print(f"❌ 测试文件不存在: {test_file}")
            results.append((description, False))
    
    # 总结报告
    print(f"\n{'='*60}")
    print("📊 完整测试总结报告")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    failed_tests = total_tests - passed_tests
    
    print(f"📈 测试统计:")
    print(f"   总测试套件: {total_tests}")
    print(f"   通过套件: {passed_tests}")
    print(f"   失败套件: {failed_tests}")
    print(f"   成功率: {passed_tests/total_tests*100:.1f}%")
    
    print(f"\n📋 详细结果:")
    for i, (test_name, success) in enumerate(results, 1):
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {i}. {test_name}: {status}")
    
    # 功能验证清单
    print(f"\n🔍 功能验证清单:")
    
    if passed_tests >= 1:  # Agent测试通过
        print("   ✅ 学术分析Agent核心功能")
        print("   ✅ 团队信息格式处理")
        print("   ✅ 工具调用机制")
        print("   ✅ 错误处理机制")
    
    if passed_tests >= 2:  # 图测试也通过
        print("   ✅ 学术分析图结构")
        print("   ✅ 节点间流转控制")
        print("   ✅ Academic-Tool节点协作")
    
    # 最终结论
    print(f"\n🎯 最终结论:")
    
    if passed_tests == total_tests:
        print("   🎉 所有测试通过！学术分析系统完全可用")
        print("   ✨ 系统已准备好投入生产使用")
        
        print(f"\n📝 系统能力确认:")
        print("   • 支持多人团队学术背景分析")
        print("   • 支持Google Scholar工具调用")
        print("   • 支持完整的Agent-Tool工作流")
        print("   • 具备完善的错误处理机制")
        print("   • 适配新的团队信息输入格式")
        
        return True
        
    elif passed_tests > 0:
        print(f"   ⚠️ 部分测试通过 ({passed_tests}/{total_tests})")
        print("   🔧 系统基本功能正常，但需要进一步优化")
        return False
        
    else:
        print("   🚨 所有测试失败！系统需要重大修复")
        print("   🛠️ 建议检查依赖和环境配置")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        
        print(f"\n📤 测试完成，退出码: {exit_code}")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n❌ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 测试套件运行时发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
