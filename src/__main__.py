"""Entry-point shim so the package can be executed with ``python -m src``."""

import asyncio

from src.main import main

if __name__ == "__main__":
    asyncio.run(main())
