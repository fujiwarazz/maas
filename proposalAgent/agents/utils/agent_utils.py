from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from typing import List
from typing import Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import RemoveMessage
from langchain_core.tools import tool
from datetime import date, timedelta, datetime
import functools
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
from langchain_openai import ChatOpenAI
import proposalAgent.tools.tool_interface as interface
from proposalAgent.model_config import TONGYI_CONFIG
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class Toolkit:
    
    _config = 