# versions.py

import platform
import pkg_resources

print(f"Python Version: {platform.python_version()}")
print("Installed Package Versions:")

for dist in sorted(pkg_resources.working_set, key=lambda d: d.project_name.lower()):
    print(f"{dist.project_name}=={dist.version}")
