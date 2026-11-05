"""Entry point: ``python main.py --help`` lists the stages."""

import sys

from cricket_project.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
