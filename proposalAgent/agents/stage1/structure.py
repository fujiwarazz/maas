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
import base64
import json

import http.client
from PyPDF2 import PdfReader
import asyncio
import sys

client = genai.Client()
def get_genai_output(prompt: str, filepath: pathlib.Path) -> str:
    pdf_b64 = base64.b64encode(filepath.read_bytes()).decode("utf-8")

    payload = {
        "model": "gemini-2.5-flash-lite",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": filepath.name,
                            "file_data": f"data:application/pdf;base64,{pdf_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    }

    conn = http.client.HTTPSConnection("poloai.top")
    headers = {
        "Authorization": "Bearer sk-B4dxzRASc8xmVLKFV9sps4kX1cN7FhnxgrU34Qz6QSLfn39t",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    conn.request("POST", "/v1/chat/completions", json.dumps(payload), headers)
    res = conn.getresponse()
    body = res.read().decode("utf-8")
    conn.close()

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body

    choices = data.get("choices", [])
    if not choices:
        return body

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return body
def get_genai_output__(prompt: str,filepath:pathlib.Path)->str:
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
        file_path = state.get("file_path") or state.get("filepath")
        if not file_path:
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
            你是一个专业的抽取pdf机器人
            ### 任务描述
            将这篇国家自然基金项目申请书中的数据进行抽取。
            包含了以下内容：
            - 申请人的个人履历,完整的相关经历和论文背景
            - 项目团队成员及其个人履历,完整的相关经历和论文背景
            - 项目申请信息(表格数据等内容)
            - 报告正文,对于报告正文部分可以做总结,但是要保留所有核心内容，包括算法核心、创新点、实验步骤等内容。项目正文部分往往包括：1、项目的立项依据(项目背景和意义)；2、项目的主要内容以及目标或拟解决的关键问题；3、拟采取的方案的可行性分析；4、本项目的特色与创新之处；5、年度计划及预期结果；6、工作基础及保障措施(工作条件、个人相关方面的研究基础和保障措施),报告正文**一定**要保证不缺少核心内容！。
            **注意**，如果有图片、公式，那么需要在内容中添加图的说明和公式及其说明。
            **同时，每个部分都需要带上来源于的page，格式为[页面x]
           
            ### 输出格式
            {{
                "research_person_info": "申请人完整信息，包括完整的论文发表信息和其他完整履历信息",
                "research_project_team_info": "团队信息，包括完整的团队成员履历信息",
                "research_project_apply_info": "申请信息，包括完整的申请信息",
                "research_report_body_summary": "报告正文"
            }}
            """
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
        # cache here，直接根据名称返回所有信息
        file_names = ["面向生命科学领域表格科学数据优化算法研究","基于知识图谱与要素化大模型的基础研究科技成果评价体系","基于图提示微调的图预训练模型迁移学习方法研究","面向领域大数据的知识图谱构建","支持下一代人工智能的开放型高质量科学数据库"]
        basic_info = first_page_text

        for name in file_names:
            if name in basic_info:
                with open(f"/Users/peelsannaw/Desktop/codes/maas/mas4proposal/data/cached/{name}.json", "r") as f:
                    full_data = json.load(f)
                research_structure = full_data.get("research_structure", "")
                research_person_info = full_data.get("research_person_info", "")
                research_project_team_info = full_data.get("research_project_team_info", "")
                research_project_apply_info = full_data.get("research_project_apply_info", "")
                research_report_body_summary = full_data.get("research_report_body_summary", "")
                return {
                    "research_structure": research_structure,
                    "research_person_info": research_person_info,
                    "research_basic_info": first_page_text,
                    "research_project_team_info": research_project_team_info,
                    "research_project_apply_info": research_project_apply_info,
                    "research_report_body_summary": research_report_body_summary
                }
      
        res = get_genai_output__(prompt,filepath)
        
        if res.startswith("```json"):
            res = res.split("```json")[1]
        if res.startswith("```"):
            res = res.split("```")[1]
        if res.endswith("```"):
            res = res.split("```")[0]
        
        res = res.strip()
        
        try:
            result_items = json.loads(res)
            print(result_items)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"问题数据: {res}")
            result_items = {
                "research_person_info": "",
                "research_project_team_info": "",
                "research_project_apply_info": "",
                "research_report_body_summary": ""
            }
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
        
        
