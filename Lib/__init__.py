"""fast and efficient EOF analysis for CDAT and Python"""
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
from errors import EofError

# Define the objects imported by imports of the form: from eof2 import *
__all__ = ["EofError"]

try:
    # Attempt to import the EofSolver object. If this is successful, add
    # the EofSolver object to the __all__ list. Also imports the EofSolver
    # alias EofNumPy for backward compatibility.
    from eofsolve import EofSolver, EofNumPy
    __all__.append("EofSolver")
    __all__.append("EofNumPy")
except ImportError:
    # If this fails the package cannot be used. An error should be
    # raised.
    raise EofError("eof2 requires NumPy.")

try:
    # Attempt to import the Eof object. If this is successful, add the
    # Eof object to the __all__ list.
    from eofwrap import Eof
    __all__.append("Eof")
except ImportError:
    # If this fails just leave the Eof object out. This allows users
    # with NumPy but no cdms2 to use the NumPy interface. 
    pass

