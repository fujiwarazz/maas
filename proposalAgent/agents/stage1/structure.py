from turtle import st
from google.genai import types
import pathlib
import time
import json
from google import genai
from google.genai import types
import pathlib
import time
from dataclasses import dataclass
import numpy as np
import requests
from datetime import datetime
import time
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from PyPDF2 import PdfReader
import asyncio
import sys

client = genai.Client()

def get_genai_output(prompt: str,filepath:pathlib.Path)->str:
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
        types.Part.from_bytes(
            data=filepath.read_bytes(),
            mime_type='application/pdf',
        ),
        prompt],
    )
    return response.text or ""

def create_structure_node():
    
    def get_structure_output_node(state):
        print(state)
        # 支持两种字段名：file_path 和 filepath
        file_path = state.get("file_path") or state.get("filepath")
        if not file_path:
            # 如果没有文件路径，返回默认的结构化信息
            return {
                "research_structure": "基于用户输入进行分析，无PDF文件提供",
                "research_person_info": "申请人信息：待分析",
                "research_project_team_info": "团队信息：待分析", 
                "research_project_apply_info": "申请信息：待分析",
                "research_report_body_summary": "报告正文：基于用户描述进行分析"
            }
        filepath = pathlib.Path(file_path)
        
        prompt = f"""
            ### 角色描述
            你是一个专业的抽取和总结pdf机器人
            ### 任务描述
            将这篇国家自然基金项目申请书中的重要数据进行抽取。
            包含了以下内容：
            - 申请人的个人履历,完整的相关经历和论文背景
            - 项目团队成员及其个人履历,完整的相关经历和论文背景
            - 项目申请信息(表格数据等内容)
            - 报告正文,对于报告正文部分可以做总结,但是要保留完整意思。项目正文部分往往包括：1、项目的立项依据(项目背景和意义)；2、项目的主要内容以及目标或拟解决的关键问题；3、拟采取的方案的可行性分析；4、本项目的特色与创新之处；5、年度计划及预期结果；6、工作基础及保障措施(工作条件、个人相关方面的研究基础和保障措施)。
            ### 输出格式
            {{
                "research_person_info": "申请人完整信息，包括完整的论文发表信息和其他完整履历信息",
                "research_project_team_info": "团队信息，包括完整的团队成员履历信息",
                "research_project_apply_info": "申请信息，包括完整的申请信息",
                "research_report_body_summary": "报告正文"
            }}
            """

        res = get_genai_output(prompt,filepath)
        print(f"DEBUG_structure_node: {res}")
        
        if res.startswith("```json"):
            res = res.split("```json")[1]
        if res.startswith("```"):
            res = res.split("```")[1]
        if res.endswith("```"):
            res = res.split("```")[0]
        
        res = res.strip()
        
        try:
            result_items = json.loads(res)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"问题数据: {res}...")
            result_items = {
                "research_person_info": "",
                "research_project_team_info": "",
                "research_project_apply_info": "",
                "research_report_body_summary": ""
            }
        

        def read_first_page_text(filepath):
            try:
                reader = PdfReader(str(filepath))
                if len(reader.pages) > 0:
                    first_page = reader.pages[1]
                    return first_page.extract_text()
                else:
                    return ""
            except Exception as e:
                print(f"读取PDF第一页失败: {e}")
                return ""

        first_page_text = read_first_page_text(filepath)

        return {
            "research_structure": res,
            "research_basic_info": first_page_text,
            "research_person_info": result_items["research_person_info"],
            "research_project_team_info": result_items["research_project_team_info"],
            "research_project_apply_info": result_items["research_project_apply_info"],
            "research_report_body_summary": result_items["research_report_body_summary"]
        }
        
    node = get_structure_output_node
    return node
        
        
if __name__ == "__main__":
    
    structure_agent = create_structure_node()
    start_time = time.time()
    result = structure_agent({"messages": [("user", "分析这篇文章")],"file_path":"/Users/peelsannaw/Desktop/提交版本.pdf"})
    end_time = time.time()
    result_items = result['research_structure'].split("============")
    print("\n=======================".join(result_items))
    
    print(f"time_cost:{end_time - start_time}")