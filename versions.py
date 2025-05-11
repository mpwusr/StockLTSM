# versions.py

import platform
import pkg_resources
import sys

# Define the output file (if redirected)
output_file = "versions.txt"

# Custom class to write to both stdout and a file
class Tee:
    def __init__(self, stdout, file):
        self.stdout = stdout
        self.file = file

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

# Open the output file and set up dual output
with open(output_file, 'w') as f:
    # Redirect stdout to both terminal and file
    sys.stdout = Tee(sys.__stdout__, f)

    # Your original code
    print("============================")
    print(f"Python Version: {platform.python_version()}")
    print("============================")
    print("Redirect to text file or stdout")
    print("Installed Package Versions:")

    for dist in sorted(pkg_resources.working_set, key=lambda d: d.project_name.lower()):
        print(f"{dist.project_name}=={dist.version}")

# Restore original stdout
sys.stdout = sys.__stdout__
