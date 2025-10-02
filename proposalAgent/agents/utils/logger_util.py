import logging
import functools
import inspect
import copy
from typing import Mapping, Iterable, Optional, Any, Dict

def log_state(
    *,
    fields: Optional[Iterable[str]] = None,   # 只输出这些字段；None=全量
    show_diff: bool = True,                   # 返回时显示增量（相对入参）
    mask_keys: Iterable[str] = ("api_key", "token", "password", "secret"),
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    label: Optional[str] = None,              # 日志前缀标识
):
    """
    用于 langgraph 节点/边函数的装饰器。
    约定：被装饰函数的第一个参数是 state（Mapping 或具备 dict 接口的对象）。
    返回值可以是原位修改后的 state，或增量 dict（langgraph 支持的更新）。
    """
    log = logger or logging.getLogger("langgraph.state")

    def _to_dict(x: Any) -> Dict[str, Any]:
        # 支持 dict / pydantic / 有 dict() 方法的对象
        if isinstance(x, Mapping):
            return dict(x)
        if hasattr(x, "dict") and callable(getattr(x, "dict")):
            return x.dict()
        if hasattr(x, "__dict__"):
            return dict(vars(x))
        return {"value": x}

    def _mask(d: Dict[str, Any]) -> Dict[str, Any]:
        def mkey(k: str) -> bool:
            lk = k.lower()
            return any(s in lk for s in mask_keys)
        def rec(v):
            if isinstance(v, Mapping):
                return {kk: rec(vv) for kk, vv in v.items()}
            if isinstance(v, (list, tuple)):
                return type(v)(rec(i) for i in v)
            if isinstance(v, str) and len(v) > 8:
                return v[:4] + "..." + v[-2:]
            return v
        out = {}
        for k, v in d.items():
            out[k] = "***" if mkey(k) else rec(v)
        return out

    def _select(d: Dict[str, Any]) -> Dict[str, Any]:
        if not fields:
            return d
        sel = {}
        for f in fields:
            # 支持点号路径：foo.bar.baz
            cur = d
            ok = True
            for part in f.split("."):
                if isinstance(cur, Mapping) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if ok:
                sel[f] = cur
        return sel

    def _diff(after: Dict[str, Any], before: Dict[str, Any]) -> Dict[str, Any]:
        # 只做浅层 diff，够用且稳
        changed = {}
        keys = set(after.keys()) | set(before.keys())
        for k in keys:
            if after.get(k) != before.get(k):
                changed[k] = {"before": before.get(k), "after": after.get(k)}
        return changed

    def _log(prefix: str, d: Dict[str, Any]):
        safe = _mask(_select(d))
        tag = f"[{label}] " if label else ""
        log.log(level, f"{tag}{prefix}: {safe}")

    def decorator(fn):
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def aw(*args, **kwargs):
                if not args:
                    return await fn(*args, **kwargs)

                before_raw = _to_dict(args[0])
                before = copy.deepcopy(before_raw)
                _log("ENTER", before)

                result = await fn(*args, **kwargs)

                # 计算“返回后的状态视角”
                # langgraph 常见两种：原位修改后的 state，或返回增量 dict
                if isinstance(result, Mapping):
                    # 增量视为对 before 的更新，合并出 after
                    after = copy.deepcopy(before)
                    for k, v in result.items():
                        after[k] = v
                else:
                    after = _to_dict(result)

                if show_diff:
                    _log("RETURN~DIFF", _diff(_select(after), _select(before)))
                else:
                    _log("RETURN", after)
                return result
            return aw
        else:
            @functools.wraps(fn)
            def w(*args, **kwargs):
                if not args:
                    return fn(*args, **kwargs)

                before_raw = _to_dict(args[0])
                before = copy.deepcopy(before_raw)
                _log("ENTER", before)

                result = fn(*args, **kwargs)

                if isinstance(result, Mapping):
                    after = copy.deepcopy(before)
                    for k, v in result.items():
                        after[k] = v
                else:
                    after = _to_dict(result)

                if show_diff:
                    _log("RETURN~DIFF", _diff(_select(after), _select(before)))
                else:
                    _log("RETURN", after)
                return result
            return w
    return decorator
