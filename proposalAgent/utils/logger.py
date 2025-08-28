import logging
import uuid
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
import contextvars

# ------------------------
# 上下文变量定义
# ------------------------
trace_id_var = contextvars.ContextVar("trace_id", default="N/A")
span_id_var = contextvars.ContextVar("span_id", default="")
parent_span_id_var = contextvars.ContextVar("parent_span_id", default="")
sw_ctx_var = contextvars.ContextVar("sw_ctx", default="N/A")


def generate_id():
    """生成短 uuid（可换成 snowflake 或 nanoid）"""
    return uuid.uuid4().hex[:12]


def set_logging_context(trace_id=None, span_id=None, parent_span_id=None, sw_ctx=None):
    """设置当前日志上下文变量，如无值自动生成"""
    trace_id_var.set(trace_id or generate_id())
    span_id_var.set(span_id or generate_id())
    parent_span_id_var.set(parent_span_id or "root")
    sw_ctx_var.set(sw_ctx or "default-op")


# ------------------------
# 自定义 LogRecord 工厂：自动注入上下文
# ------------------------
_old_factory = logging.getLogRecordFactory()


def _custom_log_record_factory(*args, **kwargs):
    """
    自定义 LogRecord 工厂，自动注入上下文变量
    :param args:
    :param kwargs:
    :return:
    """
    record = _old_factory(*args, **kwargs)
    record.trace_id = trace_id_var.get()
    record.span_id = span_id_var.get()
    record.parent_span_id = parent_span_id_var.get()
    record.sw_ctx = sw_ctx_var.get()
    return record


logging.setLogRecordFactory(_custom_log_record_factory)


class CustomLogFormatter(logging.Formatter):
    """
    自定义日志格式化器，自动注入上下文变量
    # ------------------------
    # 日志格式化器
    # ------------------------
    """
    def format(self, record):
        """
        格式化日志记录，自动注入上下文变量
        :param record: 日志记录对象
        :return: 格式化后的日志字符串
        """
        log_time = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        class_and_line = f"{os.path.basename(record.pathname)}:{record.lineno}"
        return f"[{log_time}][{record.trace_id},{record.span_id},{record.parent_span_id}][TID: {record.trace_id}] [SW_CTX: [{record.sw_ctx}]][{record.levelname}][{class_and_line}] {record.getMessage()}"


def _create_file_handler(file_path: str, formatter: logging.Formatter):
    """
    创建文件处理器，自动配置按天切割日志
    :param file_path: 日志文件路径
    :param formatter: 日志格式化器
    :return: 文件处理器
    """
    # ------------------------
    # 创建 file handler（按天切割，保留 7 天）
    # ------------------------
    handler = TimedRotatingFileHandler(
        filename=file_path,
        when='midnight',
        interval=1,
        backupCount=7,
        encoding='utf-8'
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    return handler


def get_logger(service_name: str, log_dir="../../logs"):
    """
    获取日志记录器，自动配置 console + 文件输出
    :param service_name: 服务名称，用于区分不同模块的日志
    :param log_dir: 日志目录，默认 "../logs"
    :return: 配置好的日志记录器
    """
    # ------------------------
    # 获取 logger，自动配置 console + 文件输出
    # ------------------------
    today = datetime.now().strftime("%Y-%m-%d")
    base_dir = os.path.join(log_dir, today)
    os.makedirs(base_dir, exist_ok=True)

    logger = logging.getLogger(service_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = CustomLogFormatter()

        # 控制台输出
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        # 模拟 log4j2 中多个文件 appender
        app_files = {
            "APP_LOG": "application.log",
        }

        for _, filename in app_files.items():
            file_path = os.path.join(base_dir, filename)
            file_handler = _create_file_handler(file_path, formatter)
            logger.addHandler(file_handler)

    return logger
