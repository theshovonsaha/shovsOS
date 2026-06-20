import asyncio
from plugins.tools_web import _web_search
async def main():
    print(await _web_search("apple stock"))
asyncio.run(main())
