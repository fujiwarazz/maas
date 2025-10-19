FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先复制依赖声明文件，利用 Docker 层缓存
COPY pyproject.toml requirements.txt ./

RUN pip install --upgrade pip \
    && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# 复制项目源码
COPY . .

# 默认暴露 FastAPI 端口
EXPOSE 8000

# 如果需要自定义配置，可在运行时覆盖以下命令
CMD ["uvicorn", "proposalAgent.api.server:app", "--host", "0.0.0.0", "--port", "8000"]



