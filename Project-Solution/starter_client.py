import asyncio
import json
import logging
import os
import re
import shutil
import time
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, List, Dict, TypedDict
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL = 'claude-sonnet-4-5-20250929'

# Rate limiting configuration
class RateLimiter:
    """Simple rate limiter for API calls."""
    def __init__(self, calls_per_minute: int = 15):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0

    async def wait_if_needed(self) -> None:
        """Wait if necessary to maintain rate limit."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()

rate_limiter = RateLimiter(calls_per_minute=15)

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict


def extract_tool_content(result: Any) -> str:
    """Extract text content from a tool execution result."""
    if hasattr(result, 'content'):
        return result.content[0].text if result.content else str(result)
    return str(result)


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

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
            return [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema or {}
                }
                for tool in tools_response.tools
            ]
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
                return await self.session.call_tool(tool_name, arguments, read_timeout_seconds=timedelta(seconds=15))
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
                CREATE TABLE IF NOT EXISTS pricing_plans (
                    id INTEGER PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    plan_name TEXT NOT NULL,
                    input_tokens REAL,
                    output_tokens REAL,
                    currency TEXT DEFAULT 'USD',
                    billing_period TEXT,
                    features TEXT,
                    limitations TEXT,
                    source_query TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
                max_tokens=4096,
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return next((c.text for c in response.content if c.type == 'text'), '').strip()
        except Exception as e:
            logging.error(f"Error in structured extraction: {e}")
            return '{"plans": []}'

    def _clean_json_string(self, json_str: str) -> str:
        """Clean and fix common JSON formatting issues."""
        json_str = json_str.replace("```json\n", "").replace("```json", "").replace("```", "")
        json_match = re.search(r'\{[\s\S]*\}', json_str)
        if json_match:
            json_str = json_match.group(0)
        return json_str.strip()

    def _attempt_json_parse(self, json_str: str) -> dict | None:
        """Attempt to parse JSON with multiple recovery strategies."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        try:
            fixed = re.sub(r',\s*}', '}', json_str)
            fixed = re.sub(r',\s*]', ']', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        try:
            fixed = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        return None

    @staticmethod
    def _esc(value: Any) -> str:
        """Escape single quotes for SQL string literals."""
        return str(value).replace("'", "''")

    async def extract_and_store_data(self, user_query: str, llm_response: str) -> None:
        """Extract structured data from LLM response and store it."""
        try:
            if not llm_response or len(llm_response) < 100:
                logger.info("Response too short for data extraction")
                return

            if "pricing" not in llm_response.lower() and "price" not in llm_response.lower():
                logger.info("No pricing information found in response")
                return

            llm_response_limited = llm_response[:8000]

            await rate_limiter.wait_if_needed()

            extraction_prompt = f"""Extract ALL pricing plans from this text as JSON. Return ONLY valid JSON, no other text.

Text: {llm_response_limited}

Return this exact JSON structure:
{{
    "company_name": "the company name",
    "plans": [
        {{
            "plan_name": "model or plan name",
            "input_tokens": number or null,
            "output_tokens": number or null,
            "currency": "USD",
            "billing_period": "description",
            "features": ["feature1", "feature2"],
            "limitations": "any limitations"
        }}
    ]
}}

If no pricing found, return {{"company_name": "Unknown", "plans": []}}"""

            extraction_response = await self._get_structured_extraction(extraction_prompt)
            cleaned_response = self._clean_json_string(extraction_response)
            pricing_data = self._attempt_json_parse(cleaned_response)

            if not pricing_data:
                logger.warning(f"Failed to parse JSON: {cleaned_response[:50]}")
                return

            plans = pricing_data.get("plans") or []
            if not plans:
                logger.info("No pricing plans found in extraction response")
                return

            company_name = self._esc(pricing_data.get("company_name", "Unknown"))
            source_query = self._esc(user_query)

            for plan in plans:
                try:
                    input_tokens = plan.get("input_tokens") if plan.get("input_tokens") is not None else "NULL"
                    output_tokens = plan.get("output_tokens") if plan.get("output_tokens") is not None else "NULL"

                    query = f"""
                    INSERT INTO pricing_plans
                    (company_name, plan_name, input_tokens, output_tokens, currency,
                     billing_period, features, limitations, source_query)
                    VALUES ('{company_name}', '{self._esc(plan.get("plan_name", ""))}',
                            {input_tokens}, {output_tokens},
                            '{self._esc(plan.get("currency", "USD"))}',
                            '{self._esc(plan.get("billing_period", ""))}',
                            '{self._esc(json.dumps(plan.get("features", [])))}',
                            '{self._esc(plan.get("limitations", ""))}',
                            '{source_query}')
                    """
                    await self.sqlite_server.execute_tool("write_query", {"query": query})
                except Exception as e:
                    logger.error(f"Error inserting plan {plan.get('plan_name')}: {e}")
                    continue

            logger.info(f"Stored {len(plans)} pricing plans")

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

    SYSTEM_PROMPT = (
        "You are a helpful assistant with access to web scraping and database tools. "
        "When answering questions about pricing or costs, ALWAYS check the database first "
        "by querying the pricing_plans table using read_query. "
        "Only use scrape_websites if the required data is not already stored in the database. "
        "Never re-scrape a provider that already has data in the database. "
        "The pricing_plans table has these columns: "
        "id, company_name, plan_name, input_tokens, output_tokens, currency, "
        "billing_period, features, limitations, source_query, created_at."
    )

    async def _call_claude(self, messages: list) -> Any:
        """Call Claude API with rate limiting."""
        await rate_limiter.wait_if_needed()
        return self.anthropic.messages.create(
            max_tokens=2024,
            model=MODEL,
            tools=self.available_tools,
            system=self.SYSTEM_PROMPT,
            messages=messages
        )

    async def process_query(self, query: str) -> None:
        """Process a user query and extract/store relevant data."""
        messages = [{'role': 'user', 'content': query}]
        response = await self._call_claude(messages)

        full_response = ""

        while True:
            assistant_content = []
            has_tool_use = False

            for content in response.content:
                if content.type == 'text':
                    full_response += content.text
                    assistant_content.append(content)
                    print(content.text, end="", flush=True)
                elif content.type == 'tool_use':
                    has_tool_use = True
                    assistant_content.append(content)
                    tool_name = content.name
                    tool_use_id = content.id
                    tool_input = content.input

                    print(f"\n[Calling tool: {tool_name}]", end="", flush=True)

                    try:
                        server = next((s for s in self.servers if self.tool_to_server.get(tool_name) == s.name), None)
                        if not server:
                            raise Exception(f"Tool {tool_name} not found on any server")

                        tool_result = await server.execute_tool(tool_name, tool_input)
                        tool_content = extract_tool_content(tool_result)
                        full_response += tool_content

                        messages.append({'role': 'assistant', 'content': assistant_content})
                        messages.append({
                            'role': 'user',
                            'content': [{
                                'type': 'tool_result',
                                'tool_use_id': tool_use_id,
                                'content': tool_content
                            }]
                        })

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

                    assistant_content = []

            if response.stop_reason == "end_turn" or not has_tool_use:
                break

            response = await self._call_claude(messages)

        if full_response.strip():
            print()

        if self.data_extractor and full_response.strip():
            await self.data_extractor.extract_and_store_data(query, full_response.strip())

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

            except (KeyboardInterrupt, EOFError):
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
            result = await self.sqlite_server.execute_tool("read_query", {
                "query": "SELECT company_name, plan_name, input_tokens, output_tokens, currency FROM pricing_plans ORDER BY created_at DESC LIMIT 5"
            })
            print("\n=== Recently Stored Pricing Data ===")
            raw = extract_tool_content(result)
            try:
                rows = json.loads(raw)
                for row in rows:
                    company = row.get("company_name", "Unknown")
                    plan = row.get("plan_name", "Unknown")
                    input_t = row.get("input_tokens", "N/A")
                    output_t = row.get("output_tokens", "N/A")
                    currency = row.get("currency", "USD")
                    print(f"  • {company} | {plan} | Input: {input_t} {currency}/1M | Output: {output_t} {currency}/1M")
            except (json.JSONDecodeError, TypeError):
                print(raw)
            print("=" * 40)
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
    script_dir = Path(__file__).parent
    env_file = script_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()

    config = Configuration()
    config_file = script_dir / "server_config.json"
    server_config = config.load_config(config_file)

    servers = [Server(name, srv_config) for name, srv_config in server_config["mcpServers"].items()]
    chat_session = ChatSession(servers, config.anthropic_api_key)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())
