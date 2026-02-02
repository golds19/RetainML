"""Launch the FastAPI prediction server."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "serving.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
