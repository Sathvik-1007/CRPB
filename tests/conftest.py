import os
import sys

# Ensure the repository root is on sys.path so that the `crpb` package can be imported
# during pytest collection, regardless of the working directory pytest chooses.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
