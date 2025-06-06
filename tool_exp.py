import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Any, Union, get_args, get_origin

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import FunctionToolParam
from pydantic import BaseModel, TypeAdapter

load_dotenv()


class Number(BaseModel):
    number: int


def _remove_titles_recursively(obj: dict[str, Any] | list[Any] | Any) -> None:
    """Recursively remove all 'title' fields from a nested dictionary."""
    if isinstance(obj, dict):
        # Remove title if present
        obj.pop("title", None)

        # Recursively process all values
        for value in obj.values():
            _remove_titles_recursively(value)
    elif isinstance(obj, list):
        # Process each item in lists
        for item in obj:
            _remove_titles_recursively(item)


def add(a: int, b: Number) -> int:
    """Adds two numbers together"""
    return Number(number=a + b.number).number


client = OpenAI()


class Tool:
    def __init__(self, func: Callable[..., Any]):
        self.func = func
        self.name = func.__name__
        self.is_async = inspect.iscoroutinefunction(func)

    def schema(self) -> FunctionToolParam:
        schema = TypeAdapter(self.func).json_schema()

        # Add additionalProperties: false to all objects in $defs
        if "$defs" in schema:
            for def_schema in schema["$defs"].values():
                if def_schema.get("type") == "object":
                    def_schema["additionalProperties"] = False
        _remove_titles_recursively(schema)

        return FunctionToolParam(
            name=self.name, parameters=schema, description=self.func.__doc__, type="function", strict=True
        )

    async def call(self, kwargs: dict[str, Any]) -> Any:
        """Call function with automatic argument type conversion."""
        sig = inspect.signature(self.func)
        converted_kwargs = {}

        for param_name, param in sig.parameters.items():
            if param_name not in kwargs:
                # Use default if available
                if param.default is not inspect.Parameter.empty:
                    converted_kwargs[param_name] = param.default
                continue

            value = kwargs[param_name]
            param_type = param.annotation

            try:
                converted_kwargs[param_name] = self._convert_value(value, param_type)
            except Exception as e:
                raise ValueError(f"Failed to convert parameter '{param_name}': {e}")

        return await self.func(**converted_kwargs) if self.is_async else self.func(**converted_kwargs)

    def _convert_value(self, value: Any, param_type: Any) -> Any:
        """Convert a value to the expected parameter type."""
        # Handle None/Optional
        if value is None:
            return None

        # Handle Union types (including Optional)
        origin = get_origin(param_type)
        if origin is Union:
            args = get_args(param_type)
            # Try each type in the union
            for arg_type in args:
                if arg_type is type(None):
                    continue
                try:
                    return self._convert_value(value, arg_type)
                except Exception:
                    continue
            return value

        # Handle List types
        if origin is list:
            if isinstance(value, list):
                inner_type = get_args(param_type)[0] if get_args(param_type) else Any
                return [self._convert_value(item, inner_type) for item in value]  # type: ignore

        # Handle BaseModel
        if isinstance(value, dict) and inspect.isclass(param_type) and issubclass(param_type, BaseModel):
            return param_type(**value)

        return value  # type: ignore


# Usage:
tools = {"add": Tool(add)}

res = client.responses.create(
    model="gpt-4.1-nano",
    input="Use the `sum` function to solve this: What is 6+3? ",
    tools=[tool.schema() for tool in tools.values()],
)


async def main():
    for tool_call in res.output:
        if tool_call.type != "function_call":
            continue

        name = tool_call.name
        kwargs = json.loads(tool_call.arguments)
        print(kwargs)

        if name in tools:
            result = await tools[name].call(kwargs)
            print(result)


if __name__ == "__main__":
    asyncio.run(main())
