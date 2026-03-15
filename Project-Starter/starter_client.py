import asyncio
import json
import logging
import os
import re
import shutil
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, List, Dict, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str | Path) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
            ValueError: If configuration file is missing required fields.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in configuration file: {str(e)}", e.doc, e.pos)

        if "mcpServers" not in config:
            raise ValueError("Configuration file missing required field 'mcpServers'")

        return config

    @property
    def anthropic_api_key(self) -> str:
        """Get the Anthropic API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = shutil.which("npx") if self.config["command"] == "npx" else self.config["command"]
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        # create params with command and args
        server_params = StdioServerParameters(
            command=command,
            args=self.config.get("args", [])
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.session = session
            logging.info(f"✓ Server '{self.name}' initialized")
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[ToolDefinition]:
        """List available tools from the server.

        Returns:
            A list of available tool definitions.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if self.session is None:
            raise RuntimeError(f"Server {self.name} is not initialized")

        try:
            tools_response = await self.session.list_tools()
            tools: List[ToolDefinition] = []
            for tool in tools_response.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema or {}
                })
            return tools
        except Exception as e:
            logging.error(f"Error listing tools from server {self.name}: {e}")
            raise

    async def execute_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            retries: int = 2,
            delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if self.session is None:
            raise RuntimeError(f"Server {self.name} is not initialized")

        last_error = None
        for attempt in range(retries + 1):
            try:
                logging.info(f"Executing tool {tool_name} on server {self.name} (attempt {attempt + 1}/{retries + 1})")
                result = await self.session.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                last_error = e
                if attempt < retries:
                    logging.warning(f"Tool execution failed, retrying in {delay}s: {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    logging.error(f"Tool execution failed after {retries + 1} attempts: {str(e)}")

        raise last_error if last_error else Exception(f"Failed to execute tool {tool_name}")

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class DataExtractor:
    """Handles extraction and storage of structured data from LLM responses."""

    def __init__(self, sqlite_server: Server, anthropic_client: Anthropic):
        self.sqlite_server = sqlite_server
        self.anthropic = anthropic_client

    async def setup_data_tables(self) -> None:
        """Setup tables for storing extracted data."""
        try:

            await self.sqlite_server.execute_tool("write_query", {
                "query": """
                         CREATE TABLE IF NOT EXISTS pricing_plans
                         (
                             id             INTEGER PRIMARY KEY,
                             company_name   TEXT NOT NULL,
                             plan_name      TEXT NOT NULL,
                             input_tokens   REAL,
                             output_tokens  REAL,
                             currency       TEXT     DEFAULT 'USD',
                             billing_period TEXT,
                             features       TEXT,
                             limitations    TEXT,
                             source_query   TEXT,
                             created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
                         )
                         """
            })

            logging.info("✓ Data extraction tables initialized")

        except Exception as e:
            logging.error(f"Failed to setup data tables: {e}")

    async def _get_structured_extraction(self, prompt: str) -> str:
        """Use Claude to extract structured data."""
        try:
            response = self.anthropic.messages.create(
                max_tokens=1024,
                model='claude-sonnet-4-5-20250929',
                messages=[{'role': 'user', 'content': prompt}]
            )

            text_content = ""
            for content in response.content:
                if content.type == 'text':
                    text_content += content.text

            return text_content.strip()

        except Exception as e:
            logging.error(f"Error in structured extraction: {e}")
            return '{"error": "extraction failed"}'

    async def extract_and_store_data(self, user_query: str, llm_response: str,
                                     source_url: str = None) -> None:
        """Extract structured data from LLM response and store it."""
        try:
            extraction_prompt = f"""
            Analyze this text and extract pricing information in JSON format:
            
            Text: {llm_response}
            
            Extract pricing plans with this structure:
            {{
                "company_name": "company name",
                "plans": [
                    {{
                        "plan_name": "plan name",
                        "input_tokens": number or null,
                        "output_tokens": number or null,
                        "currency": "USD",
                        "billing_period": "monthly/yearly/one-time",
                        "features": ["feature1", "feature2"],
                        "limitations": "any limitations mentioned",
                        "query": "the user's query"
                    }}
                ]
            }}
            
            Return only valid JSON, no other text. Do not return your response enclosed in ```json```
            """

            extraction_response = await self._get_structured_extraction(extraction_prompt)
            extraction_response = extraction_response.replace("```json\n", "").replace("```", "")
            pricing_data = json.loads(extraction_response)

            for plan in pricing_data.get("plans", []):
                # Insert into database
                await self.sqlite_server.execute_tool("write_query", {
                    "query": """
                             INSERT INTO pricing_plans
                             (company_name, plan_name, input_tokens, output_tokens, currency,
                              billing_period, features, limitations, source_query)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                             """,
                    "parameters": [
                        pricing_data.get("company_name", "Unknown"),
                        plan.get("plan_name", ""),
                        plan.get("input_tokens"),
                        plan.get("output_tokens"),
                        plan.get("currency", "USD"),
                        plan.get("billing_period", ""),
                        json.dumps(plan.get("features", [])),
                        plan.get("limitations", ""),
                        user_query
                    ]
                })

            logger.info(f"Stored {len(pricing_data.get('plans', []))} pricing plans")

        except Exception as e:
            logging.error(f"Error extracting pricing data: {e}")


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], api_key: str) -> None:
        self.servers: list[Server] = servers
        self.anthropic = Anthropic(api_key=api_key)
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_server: Dict[str, str] = {}
        self.sqlite_server: Server | None = None
        self.data_extractor: DataExtractor | None = None

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in reversed(self.servers):
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_query(self, query: str) -> None:
        """Process a user query and extract/store relevant data."""
        # Convert tools to the correct format for Anthropic API
        tools_for_api = []
        for tool in self.available_tools:
            tools_for_api.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"]
            })

        messages = [{'role': 'user', 'content': query}]
        response = self.anthropic.messages.create(
            max_tokens=2024,
            model='claude-sonnet-4-5-20250929',
            tools=tools_for_api,
            messages=messages
        )

        full_response = ""
        source_url = None
        used_web_search = False

        process_query = True
        while process_query:
            assistant_content = []
            for content in response.content:
                if content.type == 'text':
                    # Collect text response
                    full_response += content.text
                    assistant_content.append(content)
                    print(content.text, end="", flush=True)
                elif content.type == 'tool_use':
                    # Handle tool calls
                    assistant_content.append(content)
                    tool_name = content.name
                    tool_use_id = content.id
                    tool_input = content.input

                    print(f"\n[Calling tool: {tool_name}]", end="", flush=True)

                    try:
                        # Find which server has this tool
                        server = next((s for s in self.servers if self.tool_to_server.get(tool_name) == s.name), None)
                        if not server:
                            raise Exception(f"Tool {tool_name} not found on any server")

                        # Execute the tool
                        tool_result = await server.execute_tool(tool_name, tool_input)

                        # Extract content if it's a complex object
                        if hasattr(tool_result, 'content'):
                            tool_content = tool_result.content[0].text if tool_result.content else str(tool_result)
                        else:
                            tool_content = str(tool_result)

                        # Extract URL if available
                        if "http" in tool_content:
                            source_url = self._extract_url_from_result(tool_content)

                        if "web_search" in tool_name.lower():
                            used_web_search = True

                        full_response += tool_content

                        # Add tool result to messages
                        messages.append({'role': 'assistant', 'content': assistant_content})
                        messages.append({
                            'role': 'user',
                            'content': [{
                                'type': 'tool_result',
                                'tool_use_id': tool_use_id,
                                'content': tool_content
                            }]
                        })

                        # Get next response from Claude
                        response = self.anthropic.messages.create(
                            max_tokens=2024,
                            model='claude-sonnet-4-5-20250929',
                            tools=tools_for_api,
                            messages=messages
                        )

                    except Exception as e:
                        logger.error(f"Tool execution failed: {str(e)}")
                        error_msg = f"Error executing {tool_name}: {str(e)}"
                        messages.append({'role': 'assistant', 'content': assistant_content})
                        messages.append({
                            'role': 'user',
                            'content': [{
                                'type': 'tool_result',
                                'tool_use_id': tool_use_id,
                                'content': error_msg,
                                'is_error': True
                            }]
                        })

                        response = self.anthropic.messages.create(
                            max_tokens=2024,
                            model='claude-sonnet-4-5-20250929',
                            tools=tools_for_api,
                            messages=messages
                        )

                    # Check if there are more tool calls
                    assistant_content = []

            # Check if we should continue the loop
            if response.stop_reason == "end_turn":
                process_query = False
            else:
                # Continue processing if there are more tool calls
                has_tool_use = any(content.type == 'tool_use' for content in response.content)
                if not has_tool_use:
                    process_query = False

        if self.data_extractor and full_response.strip():
            await self.data_extractor.extract_and_store_data(query, full_response.strip(), source_url)

    def _extract_url_from_result(self, result_text: str) -> str | None:
        """Extract URL from tool result."""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, result_text)
        return urls[0] if urls else None

    async def chat_loop(self) -> None:
        """Run an interactive chat loop."""
        print("\nMCP Chatbot with Data Extraction Started!")
        print("Type your queries, 'show data' to view stored data, or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break
                elif query.lower() == 'show data':
                    await self.show_stored_data()
                    continue

                await self.process_query(query)
                print("\n")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def show_stored_data(self) -> None:
        """Show recently stored data."""
        if not self.sqlite_server:
            logger.info("No database available")
            return

        try:
            # Query the database for recently stored pricing data
            result = await self.sqlite_server.execute_tool("read_query", {
                "query": """
                         SELECT company_name,
                                plan_name,
                                input_tokens,
                                output_tokens,
                                currency,
                                billing_period,
                                features,
                                created_at
                         FROM pricing_plans
                         ORDER BY created_at DESC
                         LIMIT 20
                         """
            })

            # Extract content if it's a complex object
            if hasattr(result, 'content'):
                content = result.content[0].text if result.content else str(result)
            else:
                content = str(result)

            print("\n=== Recently Stored Pricing Data ===")
            print(content)

        except Exception as e:
            print(f"Error showing data: {e}")

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                    if "sqlite" in server.name.lower():
                        self.sqlite_server = server
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            for server in self.servers:
                tools = await server.list_tools()
                self.available_tools.extend(tools)
                for tool in tools:
                    self.tool_to_server[tool["name"]] = server.name

            print(f"\nConnected to {len(self.servers)} server(s)")
            print(f"Available tools: {[tool['name'] for tool in self.available_tools]}")

            if self.sqlite_server:
                self.data_extractor = DataExtractor(self.sqlite_server, self.anthropic)
                await self.data_extractor.setup_data_tables()
                print("Data extraction enabled")

            await self.chat_loop()

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()

    script_dir = Path(__file__).parent
    config_file = script_dir / "server_config.json"

    server_config = config.load_config(config_file)

    servers = [Server(name, srv_config) for name, srv_config in server_config["mcpServers"].items()]
    chat_session = ChatSession(servers, config.anthropic_api_key)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())
