"""
MolsControl
An automonous on-the-fly job control system for DFT geometry optimization aided by machine learning techniques.
"""

# Add imports here
from molSimplify.molscontrol import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
