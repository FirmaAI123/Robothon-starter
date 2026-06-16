import os
import sys

# make the submission root importable so `import dexsuite` works under pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
