"""OCR 引擎抽象：OcrEngine 协议 + RapidOCR(onnxruntime) 实现。"""
from typing import Protocol

import rapidocr_onnxruntime


class OcrEngine(Protocol):
    """OCR 引擎接口：输入图片字节，返回拼接后的文本。"""

    def recognize(self, image: bytes) -> str: ...


class RapidOcrEngine(OcrEngine):
    """RapidOCR(onnxruntime) 包装：懒加载模型，recognize 拼接各行文本。"""

    def __init__(self) -> None:
        self._engine = None

    def recognize(self, image: bytes) -> str:
        """识别图片并拼接各行文本；空结果返回空串。"""
        if self._engine is None:
            self._engine = rapidocr_onnxruntime.RapidOCR()
        result = self._engine(image)
        if not result:
            return ""
        # rapidocr_onnxruntime 2.x 返回 (items, elapse) 二元组；旧版直接返回 items。
        # 无文字的图片（风景照等）items 为 None——必须回空串，绝不能拿 None 去迭代
        # （真实案例：草莓照片整条解析判 error，照片链路全断）
        items = result[0] if isinstance(result, tuple) else result
        if not items:
            return ""
        return "\n".join(item[1] for item in items)


def get_ocr_engine(name: str) -> OcrEngine | None:
    """按名称返回 OCR 引擎："rapidocr" → RapidOcrEngine；"off" 或未知 → None。"""
    if name == "rapidocr":
        return RapidOcrEngine()
    return None
