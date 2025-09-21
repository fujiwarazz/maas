#!/usr/bin/env python3
"""
测试百度搜索工具的功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'proposalAgent/tools'))

from proposalAgent.tools.baidu_util import baidu_search_with_content

def test_search_with_content():
    """测试带内容解析的搜索功能"""
    print("\n=== 测试带内容解析的搜索功能 ===")
    results = baidu_search_with_content.invoke({"query": "deepseek", "num_results": 2, "depth": 1})
    
    if results:
        for i, result in enumerate(results):
            print(f"\n结果 {i+1}:")
            print(f"标题: {result.get('title', 'N/A')}")
            print(f"摘要: {result.get('abstract', 'N/A')[:100]}...")
            print(f"URL: {result.get('url', 'N/A')}")
            
            if 'url_content' in result:
                url_content = result['url_content']
                print(f"URL内容状态: {url_content.get('status', 'N/A')}")
                print(f"页面标题: {url_content.get('title', 'N/A')}...")
                print(f"页面内容: {url_content.get('content', 'N/A')}...")
                print(f"发现链接数: {len(url_content.get('links', []))}")
    else:
        print("未找到搜索结果")

def test_deep_search():
    """测试深层解析功能"""
    print("\n=== 测试深层解析功能 ===")
    results = baidu_search_with_content.invoke({"query": "deepseek", "num_results": 1, "depth": 2})
    
    if results:
        result = results[0]
        print(f"标题: {result.get('title', 'N/A')}")
        print(f"URL: {result.get('url', 'N/A')}")
        
        if 'url_content' in result:
            url_content = result['url_content']
            print(f"URL内容状态: {url_content.get('status', 'N/A')}")
            
            if 'deeper_links' in result:
                deeper_links = result['deeper_links']
                print(f"深层链接数: {len(deeper_links)}")
                
                for j, deep_link in enumerate(deeper_links[:2]):  # 只显示前2个
                    link_info = deep_link.get('link_info', {})
                    content = deep_link.get('content', {})
                    print(f"  深层链接 {j+1}: {link_info.get('text', 'N/A')}...")
                    print(f"  深层URL: {link_info.get('url', 'N/A')}")
                    print(f"  深层内容状态: {content.get('status', 'N/A')}")
    else:
        print("未找到搜索结果")

if __name__ == "__main__":
    try:
     
        test_search_with_content()
        test_deep_search()
        print("\n=== 测试完成 ===")
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
