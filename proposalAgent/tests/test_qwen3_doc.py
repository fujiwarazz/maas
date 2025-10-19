import os
import dashscope

import os
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请在此处替换您的API-KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务base_url
)

file_object = client.files.create(file=Path("/Users/peelsannaw/Desktop/codes/maas/mas4proposal/data/提交版本.pdf"), purpose="file-extract")

print(file_object.id)

import os
from openai import OpenAI, BadRequestError

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), # 如果您没有配置环境变量，请在此处替换您的API-KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

prompt = f"""
            ### 角色描述
            你是一个专业的抽取和总结pdf机器人
            ### 任务描述
            将这篇国家自然基金项目申请书中的数据进行抽取。
            包含了以下内容：
            - 申请人的个人履历,完整的相关经历和论文背景
            - 项目团队成员及其个人履历,完整的相关经历和论文背景
            - 项目申请信息(表格数据等内容)
            - 报告正文,对于报告正文部分可以做总结,但是要保留所有核心内容，包括算法核心、创新点、实验步骤等内容。项目正文部分往往包括：1、项目的立项依据(项目背景和意义)；2、项目的主要内容以及目标或拟解决的关键问题；3、拟采取的方案的可行性分析；4、本项目的特色与创新之处；5、年度计划及预期结果；6、工作基础及保障措施(工作条件、个人相关方面的研究基础和保障措施),报告正文**一定**要保证不缺少核心内容！。
            **同时，每个部分都需要带上来源于的page，格式为[页面x]
            ### 输出格式
            {{
                "research_person_info": "申请人完整信息，包括完整的论文发表信息和其他完整履历信息",
                "research_project_team_info": "团队信息，包括完整的团队成员履历信息",
                "research_project_apply_info": "申请信息，包括完整的申请信息",
                "research_report_body_summary": "报告正文"
            }}
            """
            
try:
    import time
    start = time.time()
    completion = client.chat.completions.create(
        model="qwen-doc-turbo",
        messages=[
            {'role': 'system', 'content': 'You are a helpful structure extraction assistant.'},
            {'role': 'system', 'content': f'fileid://{file_object.id}'},
            {'role': 'user', 'content': prompt}
        ],
      
    )
    end = time.time()
    data = completion.choices[0].message.content
    print(data)
    print(f"time cost: {end - start}")

except BadRequestError as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")