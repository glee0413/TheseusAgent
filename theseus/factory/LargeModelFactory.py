import asyncio
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, Optional


class Provider(Enum):
    OPENAI = auto()
    QWEN = auto()


class ModelName(Enum):
    GPT3_5 = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    QWEN_VL_PLUS = "qwen-vl-plus"
    QWEN_VL_MAX = "qwen-vl-max"


class BaseModel(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass


class OpenAIModel(BaseModel):
    def __init__(self, model_name: ModelName, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        # 初始化 OpenAI 客户端

    async def generate(self, prompt: str) -> str:
        # 使用 OpenAI API 异步生成响应
        pass


class QwenModel(BaseModel):
    def __init__(self, model_name: ModelName, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        # 初始化 Qwen 客户端

    async def generate(self, prompt: str) -> str:
        # 使用 Qwen API 异步生成响应
        pass


class ModelFactory:
    @staticmethod
    def create_model(provider: Provider, model_name: ModelName, api_key: str) -> BaseModel:
        if provider == Provider.OPENAI:
            return OpenAIModel(model_name, api_key)
        elif provider == Provider.QWEN:
            return QwenModel(model_name, api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class UnifiedModelInterface:
    def __init__(self, provider: Provider, model_name: ModelName, api_key: str):
        self.model = ModelFactory.create_model(provider, model_name, api_key)

    async def generate(self, prompt: str) -> str:
        return await self.model.generate(prompt)


# 异步使用示例
async def main():
    # 使用 OpenAI 的 GPT-3.5 模型
    openai_interface = UnifiedModelInterface(Provider.OPENAI, ModelName.GPT3_5, "your_openai_api_key")
    openai_response = await openai_interface.generate("Hello, how are you?")
    print("OpenAI response:", openai_response)

    # 使用 Qwen 的 qwen-vl-plus 模型
    qwen_interface = UnifiedModelInterface(Provider.QWEN, ModelName.QWEN_VL_PLUS, "your_qwen_api_key")
    qwen_response = await qwen_interface.generate("你好，最近如何？")
    print("Qwen response:", qwen_response)

    # 并发调用多个模型
    tasks = [
        openai_interface.generate("What's the weather like today?"),
        qwen_interface.generate("请给我讲个笑话。")
    ]
    results = await asyncio.gather(*tasks)
    print("Concurrent results:", results)


# 运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())