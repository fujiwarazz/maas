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
import asyncio
import sys
sys.path.append("/Users/peelsannaw/Desktop/codes/maas/mas4proposal")
from proposalAgent.utils.logger import get_logger
import aiohttp

@dataclass
class ProposalOutput:
    proposal_basic_info:str
    applicant_info:str
    project_team_info:str
    project_apply_info:str
    report_body_summary:str

logger = get_logger("structure_util")

client = genai.Client()
# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type((requests.exceptions.RequestException, aiohttp.ServerDisconnectedError,ConnectionError)),
# )
async def get_genai_output(prompt: str,filepath:pathlib.Path):
    client = genai.Client()
    asyncClient = client.aio
    response = await asyncClient.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
        types.Part.from_bytes(
            data=filepath.read_bytes(),
            mime_type='application/pdf',
        ),
        prompt],
    )
    return response.text

async def get_pdf_output(filepath:pathlib.Path):
    prompt = """
    ### 角色描述
    你是一个专业的抽取和总结pdf机器人
    ### 任务描述
    将这篇国家自然基金项目申请书中的重要数据进行抽取。
    包含了以下内容：
    - 申请人的个人履历,相关经历和论文背景
    - 项目团队成员及其个人履历,相关经历和论文背景
    - 项目申请信息(表格数据等内容)
    - 报告正文,对于报告正文部分可以做总结,但是要保留完整意思。项目正文部分往往包括：1、项目的立项依据(项目背景和意义)；2、项目的主要内容以及目标或拟解决的关键问题；3、拟采取的方案的可行性分析；4、本项目的特色与创新之处；5、年度计划及预期结果；6、工作基础及保障措施(工作条件、个人相关方面的研究基础和保障措施)。
    需要给出对应的出现的[页面],比如[P10]
    输出这四个部分内容同时使用: =============进行分割
    """
    
    
    time_start = time.time()

    res = await get_genai_output(prompt,filepath)
    try:
        print(f"res:{res}")
        proposal_output = ProposalOutput(**json.loads(res))
    except:
        logger.error(f"get genai output error: {res},return origin output")
        return res
    time_end = time.time()
    print(f"time cost: {time_end - time_start}")
    return proposal_output


async def main():
    filepath = pathlib.Path("/Users/peelsannaw/Desktop/提交版本.pdf")
    
    res = await get_pdf_output(filepath)
    print(res)
    

if __name__ == "__main__":
    asyncio.run(main())