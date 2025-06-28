import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents['number of parent directories to go up'] #project_root gives the absolute root directory to the current file
#Go up by N parent directories until reaching Metric3D-main
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))# insert the root in the system path, it's where python searches for modules

