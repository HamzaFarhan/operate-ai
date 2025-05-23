from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

load_dotenv()

server = MCPServerStdio(
    command="uvx",
    args=["../excel-mcp-server", "stdio"],
)
agent = Agent(model="google-gla:gemini-2.0-flash", mcp_servers=[server])
async with agent.run_mcp_servers():
    res = await agent.run(
        "create a new excel workbook with dummy data in 'dummy.xlsx'. make up thet data. one sheet. 3 cols. 5 rows"
    )
    print(res)
    
