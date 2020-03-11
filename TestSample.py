import CoreMethod
import PlotMethod


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
nAirfoil = 31
nAirfoilWake = 7
nOutflow = 51
nWake = nAirfoilWake + nOutflow - 1
iMax = 2*nAirfoil + 2*nWake - 2
jMax = 51

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
CoreMethod.MeshQuality(X, Y, iMax, jMax)

# Plot
PlotMethod.plotGrid(X, Y)

# ---------------------------------------------------------------------------- #
# 4. PERFORM SMOOTHING WITH LAPLACE
# ---------------------------------------------------------------------------- #
# Laplace smoothing
omega = 1.5
targetError = 1e-3

X, Y, residual = CoreMethod.LaplaceSmoothing(X, Y, iMax, jMax, omega, targetError)
PlotMethod.plotResidual(residual)

# Mesh Quality Check
CoreMethod.MeshQuality(X, Y, iMax, jMax)

# Plot Grid and Residual
PlotMethod.plotGrid(X, Y)

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

# ---------------------------------------------------------------------------- #