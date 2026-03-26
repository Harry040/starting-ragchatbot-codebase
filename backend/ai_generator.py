import openai
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Zhipu AI (GLM) via OpenAI-compatible API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str, base_url: str):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content and messages
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ]
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        # Get response from Zhipu AI
        response = self.client.chat.completions.create(**api_params)

        # Handle tool execution if needed
        if tool_manager and response.choices[0].finish_reason == "tool_calls":
            return self._handle_tool_execution(response, api_params, tool_manager)

        # Return direct response
        return response.choices[0].message.content

    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response (preserve system message from original request)
        system_msg = next((m for m in base_params["messages"] if m["role"] == "system"), None)

        assistant_message = {"role": "assistant", "content": initial_response.choices[0].message.content or ""}
        assistant_message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }
            for tool_call in initial_response.choices[0].message.tool_calls or []
        ]
        messages.append(assistant_message)

        # Execute all tool calls and collect results
        for tool_call in initial_response.choices[0].message.tool_calls or []:
            tool_result = tool_manager.execute_tool(
                tool_call.function.name,
                **eval(tool_call.function.arguments) if tool_call.function.arguments else {}
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        # Prepare final API call without tools (ensure system message is present)
        if system_msg and system_msg not in messages:
            messages = [system_msg] + messages

        final_params = {
            **self.base_params,
            "messages": messages
        }

        # Get final response
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content
