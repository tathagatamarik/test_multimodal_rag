"""Test package. Run: python -m unittest discover -s tests -t ."""

import logging

# Expected warnings from failure-path tests would otherwise go to logging's
# last-resort stderr handler and clutter the test report.
logging.getLogger("src").addHandler(logging.NullHandler())
logging.getLogger("src").propagate = False
