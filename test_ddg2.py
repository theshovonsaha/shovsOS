import asyncio
from plugins.tools import _search_duckduckgo
async def main():
    print(await _search_duckduckgo("apple stock", 5))
asyncio.run(main())
