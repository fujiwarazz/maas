from google import genai
from google.genai import types
import pathlib
import time
from dataclasses import dataclass
from numpy._core.defchararray import str_len
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
from openai import OpenAI
import asyncio
import sys
sys.path.append("/Users/peelsannaw/Desktop/codes/maas/mas4proposal")
from proposalAgent.utils.logger import get_logger
import aiohttp
from pathlib import Path


logger = get_logger("structure_util")

client = OpenAI(
    api_key = "sk-gJrVzbTcTtitntvY5sdNE2tMHdM2O8AH8j9l5q48TV3gJNkh",
    base_url = "https://api.moonshot.cn/v1",
)
# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type((requests.exceptions.RequestException, aiohttp.ServerDisconnectedError,ConnectionError)),
# )
async def get_genai_output(prompt: str,filepath:pathlib.Path):
   # xlnet.pdf 是一个示例文件, 我们支持 pdf, doc 以及图片等格式, 对于图片和 pdf 文件，提供 ocr 相关能力
    file_object = client.files.create(file=filepath, purpose="file-extract")
    file_content = client.files.content(file_id=file_object.id).text
    sys_prmopt = """  ### 角色描述
    你是一个专业的抽取和总结pdf机器人
    ### 任务描述
    对家自然基金项目申请书中的重要数据进行抽取。
    回答应该包含以下内容：
    - 项目申请书基本信息，一般位于首页，要保证内容充足,申请代码可能有多个不能丢失
    - 申请人的个人履历,相关经历和论文背景
    - 项目团队成员及其个人履历,相关经历和论文背景
    - 项目申请信息(表格数据等内容)
    - 报告正文,对于报告正文部分可以做总结,但是要保留完整意思。项目正文部分往往包括：1、项目的立项依据(项目背景和意义)；2、项目的主要内容以及目标或拟解决的关键问题；3、拟采取的方案的可行性分析；4、本项目的特色与创新之处；5、年度计划及预期结果；6、工作基础及保障措施(工作条件、个人相关方面的研究基础和保障措施)。
    需要给出对应的出现的[页面],比如[P10]
    输出这五个部分内容同时使用: =================== 进行分割
    """
    messages = [
        {
            "role": "system",
            "content": sys_prmopt,
        },
        {
            "role": "user",
            "content": file_content,
        },
    ]
    completion = client.chat.completions.create(
    model="kimi-k2-0905-preview",
    messages=messages,
    temperature=0.0,
    )
    
    print(completion.choices[0].message)
    return completion.choices[0].message.content

async def get_pdf_output(filepath:pathlib.Path):
    prompt = """
    ### 角色描述
    你是一个专业的抽取和总结pdf机器人
    ### 任务描述
    将这篇国家自然基金项目申请书中的重要数据进行抽取。
    包含了以下内容：
    - 项目申请书基本信息，一般位于首页，要保证内容充足
    - 申请人的个人履历,相关经历和论文背景
    - 项目团队成员及其个人履历,相关经历和论文背景
    - 项目申请信息(表格数据等内容)
    - 报告正文,对于报告正文部分可以做总结,但是要保留完整意思。项目正文部分往往包括：1、项目的立项依据(项目背景和意义)；2、项目的主要内容以及目标或拟解决的关键问题；3、拟采取的方案的可行性分析；4、本项目的特色与创新之处；5、年度计划及预期结果；6、工作基础及保障措施(工作条件、个人相关方面的研究基础和保障措施)。
    需要给出对应的出现的[页面],比如[P10]
    输出这五个部分内容同时使用: =================== 进行分割
    """
    
    time_start = time.time()

    res = await get_genai_output(prompt,filepath)
    try:
        print(f"res:{res}")
        return res
    except:
        logger.error(f"get genai output error: {res},return origin output")
        return res
    time_end = time.time()
    print(f"time cost: {time_end - time_start}")
    return proposal_output


async def main():
    filepath = pathlib.Path("/Users/peelsannaw/Desktop/提交版本.pdf")
    res = await get_pdf_output(filepath)
    with open("res.txt", "w") as f:
        f.write(res)
    
    
if __name__ == "__main__":
    asyncio.run(main())