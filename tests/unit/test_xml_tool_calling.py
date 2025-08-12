"""
Test for XML tool call format support added for Qwen and other ChatML-based models
"""

from mlx_server_nano.model_manager.tool_calling import parse_tool_calls, has_tool_calls


class TestXMLToolCallFormat:
    """Test parsing of XML-style tool calls from ChatML models"""

    def test_simple_xml_tool_call(self):
        """Test parsing a simple XML tool call"""
        response = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'

        tool_calls = parse_tool_calls(response)
        assert len(tool_calls) == 1

        call = tool_calls[0]
        assert call.type == "function"
        assert call.function["name"] == "get_weather"
        assert '"city": "Paris"' in call.function["arguments"]

    def test_xml_tool_call_with_qwen_model(self):
        """Test XML tool call parsing with Qwen model name"""
        response = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}</tool_call>'

        # Should prefer XML format for Qwen models
        tool_calls = parse_tool_calls(
            response, "mlx-community/Qwen3-30B-A3B-4bit-DWQ-10072025"
        )
        assert len(tool_calls) == 1

        call = tool_calls[0]
        assert call.function["name"] == "get_weather"
        assert '"city": "Paris"' in call.function["arguments"]
        assert '"unit": "celsius"' in call.function["arguments"]

    def test_xml_tool_call_with_context(self):
        """Test XML tool call parsing with surrounding context (like think tags)"""
        response = """<think>
Let me check the weather for Paris in Celsius using the get_current_weather function.
</think>

<tool_call>
{"name": "get_current_weather", "arguments": {"city": "Paris", "unit": "celsius"}}
</tool_call>"""

        tool_calls = parse_tool_calls(response, "qwen-model")
        assert len(tool_calls) == 1

        call = tool_calls[0]
        assert call.function["name"] == "get_current_weather"
        assert '"city": "Paris"' in call.function["arguments"]
        assert '"unit": "celsius"' in call.function["arguments"]

    def test_multiple_xml_tool_calls(self):
        """Test parsing multiple XML tool calls"""
        response = """<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>

<tool_call>
{"name": "get_time", "arguments": {"timezone": "UTC"}}
</tool_call>"""

        tool_calls = parse_tool_calls(response)
        assert len(tool_calls) == 2

        assert tool_calls[0].function["name"] == "get_weather"
        assert tool_calls[1].function["name"] == "get_time"

    def test_xml_tool_call_complex_arguments(self):
        """Test XML tool call with complex nested arguments"""
        response = """<tool_call>
{"name": "complex_function", "arguments": {"data": {"nested": {"value": 42, "array": [1, 2, 3]}}, "settings": {"enabled": true}}}
</tool_call>"""

        tool_calls = parse_tool_calls(response)
        assert len(tool_calls) == 1

        call = tool_calls[0]
        assert call.function["name"] == "complex_function"
        # Verify the JSON is valid by checking it contains expected parts
        args = call.function["arguments"]
        assert '"nested"' in args
        assert '"value": 42' in args
        assert '"array": [1, 2, 3]' in args
        assert '"enabled": true' in args

    def test_xml_tool_call_malformed_json(self):
        """Test handling of malformed JSON in XML tool call"""
        response = (
            '<tool_call>{"name": "broken", "arguments": {invalid json}}</tool_call>'
        )

        # Should handle gracefully and return empty list
        tool_calls = parse_tool_calls(response)
        assert len(tool_calls) == 0

    def test_has_tool_calls_xml_format(self):
        """Test has_tool_calls function with XML format"""
        response_with_tool = '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        response_without_tool = "Just a regular response without tools"

        assert has_tool_calls(response_with_tool)
        assert not has_tool_calls(response_without_tool)

    def test_model_specific_optimization(self):
        """Test that model name affects parsing priority"""
        xml_response = '<tool_call>{"name": "test_func", "arguments": {"param": "value"}}</tool_call>'
        mistral_response = '[TOOL_CALLS]test_func[ARGS]{"param": "value"}'

        # Qwen model should handle XML efficiently
        qwen_xml = parse_tool_calls(xml_response, "qwen-model")
        assert len(qwen_xml) == 1

        # Mistral model should handle Mistral format efficiently
        mistral_mistral = parse_tool_calls(mistral_response, "mistral-7b")
        assert len(mistral_mistral) == 1

        # Cross-compatibility: should still work with wrong format
        qwen_mistral = parse_tool_calls(mistral_response, "qwen-model")
        mistral_xml = parse_tool_calls(xml_response, "mistral-model")
        assert len(qwen_mistral) == 1  # Fallback should work
        assert len(mistral_xml) == 1  # Fallback should work

    def test_real_qwen_response_format(self):
        """Test the exact format that caused the original issue"""
        # This is the actual response from the logs that wasn't being detected
        response = """<think>
Okay, the user is asking for the weather in Paris in Celsius. Let me check the tools provided. There's a function called get_current_weather that takes city and unit parameters. The city here is Paris, and the unit should be celsius. I need to make sure the arguments are correctly formatted. The required parameters are both present, so I can call the function with those values.
</think>

<tool_call>
{"name": "get_current_weather", "arguments": {"city": "Paris", "unit": "celsius"}}
</tool_call>"""

        tool_calls = parse_tool_calls(
            response, "mlx-community/Qwen3-30B-A3B-4bit-DWQ-10072025"
        )
        assert len(tool_calls) == 1

        call = tool_calls[0]
        assert call.function["name"] == "get_current_weather"
        assert '"city": "Paris"' in call.function["arguments"]
        assert '"unit": "celsius"' in call.function["arguments"]

        # Also test detection
        assert has_tool_calls(response, "mlx-community/Qwen3-30B-A3B-4bit-DWQ-10072025")
