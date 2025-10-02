import requests
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain.tools import tool
from baidusearch.baidusearch import search
import logging
import time

# 设置日志
logger = logging.getLogger(__name__)

class BaiduSearchUtil:
    """百度搜索工具类，支持多层级URL内容解析"""
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _extract_content_from_url(self, url: str) -> Dict[str, Any]:
        """从URL提取内容"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 移除脚本和样式标签
            for script in soup(["script", "style"]):
                script.decompose()
            
            # 提取标题
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # 提取主要内容
            content_selectors = ['article', 'main', '.content', '#content', '.main-content']
            content = ""
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(separator='\n', strip=True)
                    break
            
            # 如果没有找到主要内容，提取body内容
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text(separator='\n', strip=True)
            
            # 清理内容
            content = re.sub(r'\n\s*\n', '\n', content)
            content = content[:2000]  # 限制长度
            
            # 提取链接
            links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href:
                    link_url = urljoin(url, str(href))
                    link_text = link.get_text().strip()
                    if link_text and self._is_valid_url(link_url):
                        links.append({
                            'url': link_url,
                            'text': link_text[:100]  # 限制链接文本长度
                        })
            
            return {
                'title': title_text,
                'content': content,
                'links': links[:20],  # 限制链接数量
                'status': 'success'
            }
            
        except requests.RequestException as e:
            logger.error("解析URL %s 时出错: %s", url, str(e))
            return {
                'title': '',
                'content': f'解析失败: {str(e)}',
                'links': [],
                'status': 'error'
            }
        except Exception as e:
            logger.error("解析URL %s 时出现未知错误: %s", url, str(e))
            return {
                'title': '',
                'content': f'解析失败: {str(e)}',
                'links': [],
                'status': 'error'
            }
    
    def _is_valid_url(self, url: str) -> bool:
        """检查URL是否有效"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False
    
    def _process_search_results(self, results: List[Dict], depth: int = 0) -> List[Dict]:
        """处理搜索结果，根据深度解析URL内容"""
        processed_results = []
        
        for result in results:
            processed_result = result.copy()
            
            if depth > 0 and 'url' in result:
                logger.info("正在解析URL (深度 %d): %s", depth, result['url'])
                url_content = self._extract_content_from_url(result['url'])
                processed_result['url_content'] = url_content
                
                # 如果需要更深层次的解析
                if depth > 1 and url_content['status'] == 'success' and url_content['links']:
                    processed_result['deeper_links'] = []
                    
                    for link in url_content['links'][:3]:
                        logger.info("正在解析深层链接 (深度 %d): %s", depth-1, link['url'])
                        deeper_content = self._extract_content_from_url(link['url'])
                        processed_result['deeper_links'].append({
                            'link_info': link,
                            'content': deeper_content
                        })
                        time.sleep(1)  #
                
                time.sleep(1)  # 避免请求过快
            
            processed_results.append(processed_result)
        
        return processed_results

# 全局实例
baidu_util = BaiduSearchUtil()

@tool
def baidu_search_with_content(
    query: str, 
  #  num_results: int = 5, 
    depth: int = 0
) -> List[Dict[str, Any]]:
    """
    百度搜索工具，支持多层级URL内容解析
    
    Args:
        query (str): 搜索关键词
        depth (int): URL解析深度
            - 0: 只返回搜索结果，不解析URL内容
            - 1: 解析搜索结果中的URL内容
            - 2+: 递归解析URL内部的链接内容
    
    Returns:
        List[Dict]: 搜索结果列表，每个结果包含:
            - title: 标题
            - abstract: 摘要
            - url: 链接
            - rank: 排名
            - url_content: URL内容 (当depth > 0时)
                - title: 页面标题
                - content: 页面主要内容
                - links: 页面内链接
                - status: 解析状态
            - deeper_links: 深层链接内容 (当depth > 1时)
    
    Example:
        # 基础搜索
        results = baidu_search_with_content("人工智能", num_results=5)
        
        # 搜索并解析URL内容
        results = baidu_search_with_content("深度学习", num_results=3, depth=1)
        
        # 多层级解析
        results = baidu_search_with_content("机器学习", num_results=2, depth=2)
    """
    try:
        logger.info("开始百度搜索: %s, 结果数量: %d, 解析深度: %d", query, 3, depth)
        
        # 执行百度搜索
        raw_results = search(query, num_results=3)
        
        if not raw_results:
            logger.warning("搜索 '%s' 未返回任何结果", query)
            return []
        
        # 处理搜索结果
        processed_results = baidu_util._process_search_results(raw_results, depth)
        
        logger.info("搜索完成，返回 %d 条结果", len(processed_results))
        return processed_results
        
    except Exception as e:
        logger.error("百度搜索出错: %s", str(e))
        return [{"error": f"搜索失败: {str(e)}"}]
