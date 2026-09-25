"""
AirTouchOS entry point.

    python main.py                 # run with defaults
    python main.py --calibrate     # show live threshold read-out
    python main.py --help          # all options
"""

import sys

from airtouchos.app import main

if __name__ == "__main__":
    sys.exit(main())
