#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import asyncio
from unittest.mock import AsyncMock
from unittest.mock import Mock, patch
from theseus.factory.LargeModelFactory import Provider, ModelName, ModelFactory, UnifiedModelInterface, OpenAIModel, QwenModel

class TestModelFactory(unittest.TestCase):
    def test_create_model_openai(self):
        model = ModelFactory.create_model(Provider.OPENAI, ModelName.GPT3_5, "test_key")
        self.assertIsInstance(model, OpenAIModel)

    def test_create_model_qwen(self):
        model = ModelFactory.create_model(Provider.QWEN, ModelName.QWEN_VL_PLUS, "test_key")
        self.assertIsInstance(model, QwenModel)

    def test_create_model_unsupported(self):
        with self.assertRaises(ValueError):
            ModelFactory.create_model("unsupported", ModelName.GPT3_5, "test_key")

class TestUnifiedModelInterface(unittest.IsolatedAsyncioTestCase):
    # async def test_generate_openai(self):
    #     with patch('theseus.factory.LargeModelFactory.OpenAIModel') as mock_openai:
    #         mock_openai.return_value.generate.return_value = "OpenAI response"
    #         interface = UnifiedModelInterface(Provider.OPENAI, ModelName.GPT3_5, "test_key")
    #         response = await interface.generate("Test prompt")
    #         self.assertEqual(response, "OpenAI response")
    #         mock_openai.return_value.generate.assert_called_once_with("Test prompt")
    async def test_generate_openai(self):
        with patch('theseus.factory.LargeModelFactory.OpenAIModel') as mock_openai_class:
            # 创建一个异步模拟对象
            mock_openai_instance = AsyncMock()
            mock_openai_class.return_value = mock_openai_instance

            # 设置异步模拟对象的 generate 方法返回一个协程. 1. 设置一个返回的异步对象，然后再将异步对象的值设置一个字符串
            mock_openai_instance.generate.return_value = asyncio.Future()
            mock_openai_instance.generate.return_value.set_result("OpenAI response")

            interface = UnifiedModelInterface(Provider.OPENAI, ModelName.GPT3_5, "test_key")
            response = await interface.generate("Test prompt")
            print(f'wait response: ## type{ type(response) }{response}')
            print(f'response.result type {response.result} value {response.result()}')
            self.assertEqual(response.result(), "OpenAI response")
            mock_openai_instance.generate.assert_called_once_with("Test prompt")

    async def test_generate_qwen(self):
        with patch('theseus.factory.LargeModelFactory.QwenModel') as mock_qwen:
            mock_qwen_instance = AsyncMock()
            mock_qwen.return_value = mock_qwen_instance

            #mock_qwen.return_value.generate.return_value = "Qwen response"

            mock_qwen_instance.generate.return_value = "Qwen response"

            interface = UnifiedModelInterface(Provider.QWEN, ModelName.QWEN_VL_PLUS, "test_key")
            response = await interface.generate("测试提示")
            self.assertEqual(response, "Qwen response")
            mock_qwen.return_value.generate.assert_called_once_with("测试提示")

class TestOpenAIModel(unittest.IsolatedAsyncioTestCase):
    async def test_generate(self):
        model = OpenAIModel(ModelName.GPT3_5, "test_key")
        with patch.object(model, 'generate', return_value="Mocked OpenAI response") as mock_generate:
            response = await model.generate("Test prompt")
            self.assertEqual(response, "Mocked OpenAI response")
            mock_generate.assert_called_once_with("Test prompt")

class TestQwenModel(unittest.IsolatedAsyncioTestCase):
    async def test_generate(self):
        model = QwenModel(ModelName.QWEN_VL_PLUS, "test_key")
        with patch.object(model, 'generate', return_value="Mocked Qwen response") as mock_generate:
            response = await model.generate("测试提示")
            self.assertEqual(response, "Mocked Qwen response")
            mock_generate.assert_called_once_with("测试提示")

if __name__ == '__main__':
    unittest.main()