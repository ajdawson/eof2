"""Fast and efficient EOF analysis for Python."""
# (c) Copyright 2010-2012 Andrew Dawson. All Rights Reserved.
#     
# This file is part of eof2.
# 
# eof2 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# eof2 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
# 
# You should have received a copy of the GNU General Public License
# along with eof2.  If not, see <http://www.gnu.org/licenses/>.
from errors import EofError, EofToolError


# Define the objects imported by imports of the form: from eof2 import *
__all__ = ["EofError", "EofToolError"]

# Package version number.
__version__ = "1.5.1"

try:
    # Attempt to import the NumPy-only solver interfaces. These interfaces
    # are required to use the software so importing them should not fail.
    from eofsolve import EofSolver, EofNumPy
    __all__.append("EofSolver")
    __all__.append("EofNumPy")
    from eofmultisolve import MultipleEofSolver
    __all__.append("MultipleEofSolver")
    # Also import the NumPy-only versions of the supplementary tools.
    import nptools
    __all__.append("nptools")
except ImportError:
    # If this fails the package cannot be used. An error should be raised.
    raise EofError("eof2 requires NumPy.")

try:
    # Attempt to import the cdms2-enabled interfaces. These are not required,
    # and are ignored if importing them fails, assuming the user does not have
    # the cdms2 module available.
    from eofwrap import Eof
    __all__.append("Eof")
    from eofmultiwrap import MultipleEof
    __all__.append("MultipleEof")
    # Also import the cdms2 wrapped versions of the supplementary tools.
    import tools
    __all__.append("tools")
except ImportError:
    # If this fails just leave the Eof object out. This allows users
    # with NumPy but no cdms2 to use the NumPy interface. 
    pass

