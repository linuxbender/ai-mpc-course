from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import asyncio

from pydantic import AnyUrl

server_params = StdioServerParameters(
    command="uv",
    args=["run", "mcp", "run", "server.py"],
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            prompts = await session.list_prompts()
            print(f"Available prompts: {[p.name for p in prompts.prompts]}")

            if prompts.prompts:
                prompt = await session.get_prompt("summarize_notes", arguments={"name": "March 14, 2026"})
                print(f"Prompt result: {prompt.messages[0].content}")

            resources = await session.list_resources()
            print(f"Available resources: {[r.uri for r in resources.resources]}")

            if resources.resources:
                resource_content = await session.read_resource(resources.resources[0].uri)
                content_block = resource_content.contents[0]

                if isinstance(content_block, types.TextContent):
                    print(f"Resource content: {content_block.text}")

            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools.tools]}")

            result = await session.call_tool("add_note", arguments={"name": "March 15, 2026", "content": "Meeting with the team at 10 AM"})
            result_unstructured = result.content[0]
            if isinstance(result_unstructured, types.TextContent):
                print(f"Tool content: {result_unstructured.text}")
            if result.structuredContent:
                print(f"Structured content: {result.structuredContent}")

def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()