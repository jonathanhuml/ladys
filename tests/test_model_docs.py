import subprocess
import sys
from pathlib import Path


def test_generated_model_docs_are_current():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "website/tools/generate_model_docs.py", "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
