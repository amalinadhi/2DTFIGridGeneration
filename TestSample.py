import os

import core.CoreMethod as CoreMethod
import core.PlotMethod as PlotMethod


# ---------------------------------------------------------------------------- #
# 1. INPUT 
# ---------------------------------------------------------------------------- #
# Boundary configurations
chord = 1
rOutflow = 10*chord
rFarfield = 5*chord
wakeChord = 0.2*chord
thickness = 0.12

# Nodal configurations
jMax = 31
nAirfoil = 21
nAirfoilWake = 7
nOutflow = 51
#jMax = 101
#nAirfoil = 21
#nAirfoilWake = 7
#nOutflow = 51

# Configuration
# discretization
nWake = nAirfoilWake + nOutflow - 1
iMax = 2*nAirfoil + 2*nWake - 2

# Directory
# Create directory
dirName = "results"

if not os.path.exists(dirName):
    os.mkdir(dirName)
else:
    pass

# ---------------------------------------------------------------------------- #
# 2. INITIALIZATION OF X AND Y
# ---------------------------------------------------------------------------- #
# Initialize X and Y from the following configuration
X, Y = CoreMethod.Initialization(chord, rOutflow, rFarfield, wakeChord, 
                                    thickness, nAirfoil, nAirfoilWake, nOutflow, 
                                    nWake, iMax, jMax)

# ---------------------------------------------------------------------------- #
# 3. GENERATE XI AND ETA
# ---------------------------------------------------------------------------- #
# Boundary normalization
u, v = CoreMethod.BoundaryNormalization(X, Y, iMax, jMax)

# Boundary-blended control function
xi, eta = CoreMethod.BoundaryBlendedControlFunction(u, v, iMax, jMax)

# ---------------------------------------------------------------------------- #
# 3. PERFORM TRANSFINITE INTERPOLATION
# ---------------------------------------------------------------------------- #
# Transfinite interpolation
X, Y = CoreMethod.TFI(X, Y, xi, eta, iMax, jMax)

# Mesh Quality Check
skewnessBefore = CoreMethod.MeshQuality(X, Y, iMax, jMax, 
                                        "Mesh Quality - Before Smoothing")

# Plot
PlotMethod.plotGrid(X, Y, "Transfinite Interpolation - Before Smoothing")

# ---------------------------------------------------------------------------- #
# 4. PERFORM SMOOTHING WITH LAPLACE
# ---------------------------------------------------------------------------- #
# Laplace smoothing
omega = 1.5
targetError = 1e-3

X, Y, residual = CoreMethod.LaplaceSmoothing(X, Y, iMax, jMax, omega, targetError)
PlotMethod.plotResidual(residual)

# Mesh Quality Check
skewnessAfter = CoreMethod.MeshQuality(X, Y, iMax, jMax, 
                                        "Mesh Quality - After Smoothing")

# Plot Grid and Residual
PlotMethod.plotGrid(X, Y, "Transfinite Interpolation - After Smoothing")

# ---------------------------------------------------------------------------- #
# 5. CREATE DATA STRUCTURES
# ---------------------------------------------------------------------------- #
# Nodes coordinates
nodes = CoreMethod.NodesCoordinates(X, Y, iMax, jMax)

# Cell connectivity
# cell number
cellNumber = CoreMethod.CellNumber(X, Y, iMax, jMax)

# cell neighbor
cellNeighbor = CoreMethod.CellNeighbor(X, Y, cellNumber, iMax, jMax)

# cell nodal number
cellNodalNumber = CoreMethod.CellNodalNumber(X, Y, iMax, jMax)

# cell types
cellType = CoreMethod.CellTypes(cellNodalNumber, iMax, jMax)

# Boundary flag numbers
boundaryFlags = CoreMethod.BoundaryFlags(nOutflow, iMax, jMax)

# write data structures
CoreMethod.WriteDataStructures(nodes, cellNeighbor, cellNodalNumber, cellType,
                                boundaryFlags, iMax, jMax)

# compare mesh quality datas
PlotMethod.plotQualityComparison(skewnessBefore, skewnessAfter)

# ---------------------------------------------------------------------------- #
