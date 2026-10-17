import subprocess
import sys

steps = [
    [sys.executable, "-m", "src.train"],
    [sys.executable, "-m", "src.detect"],
    [sys.executable, "-m", "src.visualize"],
]

for step in steps:
    subprocess.run(step, check=True)

print("\nPipeline completed successfully.")

