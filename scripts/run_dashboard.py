"""Launch the Streamlit dashboard."""

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
dashboard_path = project_root / "src" / "dashboard" / "app.py"

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", "8501",
    ])
