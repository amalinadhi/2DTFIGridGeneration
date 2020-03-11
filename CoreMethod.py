import math
import numpy as np

from AirfoilEquation import NACA0012


# --------------------------------------------------------------------------------- #

def BoundaryBlendedControlFunction(u, v, iMax, jMax):
    # Boundary-Blended Control Functions
    for i in range(iMax-1):
        for j in range(jMax-1):
            part1 = (1-v[0,j])*u[i,0] + v[0,j]*u[i,jMax-1]
            part2 = 1 - (u[i,jMax-1]-u[i,0])*(v[iMax-1,j]-v[0,j])
            u[i,j] = part1/part2

            part1 = (1-u[i,0])*v[0,j] + u[i,0]*v[iMax-1,j]
            part2 = 1 - (v[iMax-1,j]-v[0,j])*(u[i,jMax-1]-u[i,0])
            v[i,j] = part1/part2

    return (u, v)

def BoundaryFlags(nOutflow, iMax, jMax):
    dataSolid = []
    dataInlet = []
    dataOutlet = []
    dataFarfield = []
    dataSymmetric = []
    dataPeriodic = []

    # for top | the whole top is farfield
    for i in range(iMax-1):
        index = (iMax-1)*(jMax-1) - (iMax-1) + i + 1
        dataFarfield.append(index)

    # for left and right | the whole left and right is outflow
    for j in range(jMax-1):
        # Left
        index = 1 + (j)*(iMax-1)
        dataOutlet.append(index)

        # Right
        index = (j+1)*(iMax-1)
        dataOutlet.append(index)

    # for bottom
    for i in range(iMax-1):
        # bottom outflow
        if (i<nOutflow):
            pass
        elif (i>((iMax-1)-(nOutflow-1)-1)):
            pass
        else:
            index = i+1
            dataSolid.append(index)

    dataFlags = [dataSolid, dataInlet, dataOutlet,
                    dataFarfield, dataSymmetric, dataPeriodic]

    return dataFlags

def BoundaryNormalization(X, Y, iMax, jMax):
    # Normalization at boundary
    meshLength = np.zeros(shape=(iMax-1, jMax-1))
    u = np.zeros(shape=(iMax, jMax))
    v = np.zeros(shape=(iMax, jMax))

    # Bottom
    totalLength = 0
    for i in range(iMax-1):
        dx = X[i+1,0] - X[i,0]
        dy = Y[i+1,0] - Y[i,0]
        dLength = math.sqrt(dx**2 + dy**2)
        totalLength = totalLength + dLength
        meshLength[i,0] = totalLength

    for i in range(iMax-1):
        u[i+1,0] = meshLength[i,0]/totalLength
        v[i+1,0] = 0

    # Top
    totalLength = 0
    for i in range(iMax-1):
        dx = X[i+1,jMax-1] - X[i,jMax-1]
        dy = Y[i+1,jMax-1] - Y[i,jMax-1]
        dLength = math.sqrt(dx**2 + dy**2)
        totalLength = totalLength + dLength
        meshLength[i,jMax-2] = totalLength
        
    for i in range(iMax-1):
        u[i+1,jMax-1] = meshLength[i,jMax-2]/totalLength
        v[i+1,jMax-1] = 1

    # reset
    meshLength = np.zeros(shape=(iMax-1, jMax-1))

    # Left
    totalLength = 0
    for i in range(jMax-1):
        dx = X[0,i+1] - X[0,i]
        dy = Y[0,i+1] - Y[0,i]
        dLength = math.sqrt(dx**2 + dy**2)
        totalLength = totalLength + dLength
        meshLength[0,i] = totalLength

    for i in range(jMax-1):
        u[0,i+1] = 0
        v[0,i+1] = meshLength[0,i]/totalLength
        
    # Right
    totalLength = 0
    for i in range(jMax-1):
        dx = X[iMax-1,i+1] - X[iMax-1,i]
        dy = Y[iMax-1,i+1] - Y[iMax-1,i]
        dLength = math.sqrt(dx**2 + dy**2)
        totalLength = totalLength + dLength
        meshLength[iMax-2,i] = totalLength

    for i in range(jMax-1):
        u[iMax-1,i+1] = 1
        v[iMax-1,i+1] = meshLength[iMax-2,i]/totalLength

    return (u, v)

def CellNeighbor(X, Y, cellNumber, iMax, jMax):
    # Cell neighbor
    nPoints = iMax*jMax
    nGhostPoints = 2*(iMax + jMax)
    nTotalPoints = nPoints + nGhostPoints
    nInnerCell = (iMax-1)*(jMax-1)
    nGhostCell = 2*((iMax-1) + (jMax-1))
    nTotalCell = nInnerCell+nGhostCell

    cellNeighboring = []

    # Cell neighbor
    # Internal: general
    for j in range(jMax-1):
        for i in range(iMax-1):
            center = int(cellNumber[i+j*(iMax-1),0])

            left = center - 1
            bottom = center - (iMax-1)
            right = center + 1
            top = center + (iMax-1)

            cellNeighboring.append([left, bottom, right, top])

    # Internal: correction: bottom-top
    for i in range(iMax-1):
        # bottom
        cellNeighboring[i][1] = nInnerCell+i+1
        # top
        cellNeighboring[nInnerCell-i-1][3] = nInnerCell + (iMax-1) + (jMax-1) + i + 1

    # Internal: correction: left-right
    for j in range(jMax-1):
        # left
        cellNeighboring[(j+1)*(iMax-1) - 1][2] = nInnerCell + (iMax-1) + j + 1
        
        # right
        cellNeighboring[1 + j*(iMax-1)-1][0] = nTotalCell-j

    # Ghost: bottom
    for i in range(iMax-1):
        center = int(cellNumber[i+nInnerCell,0])
        
        left = center - 1
        bottom = -1
        right = center + 1
        top = i+1
        
        if (i==0):
            left = -1
        elif (i==iMax-2):
            right = -1
            

        cellNeighboring.append([left, bottom, right, top])

    # Ghost: right
    for j in range(jMax-1):
        center = int(cellNumber[j+nInnerCell+2*(iMax-1)-1,0])

        left = (j+1)*(iMax-1)
        bottom = center - (iMax-1)
        right = -1
        top = center - (jMax-1)

        if (j==0):
            bottom = -1
        elif (j==(jMax-2)):
            top = -1

        cellNeighboring.append([left, bottom, right, top])

    # Ghost: up
    for i in range(iMax-1):
        center = int(cellNumber[i+nInnerCell+(iMax-1)+(jMax-1),0])
        
        left = center + 1
        bottom = nInnerCell - i
        right = center - 1
        top = -1
        
        if (i==0):
            right = -1
        elif (i==iMax-2):
            left = -1
            

        cellNeighboring.append([left, bottom, right, top])

    # Ghost: left
    for j in range(jMax-1):
        center = int(cellNumber[nTotalCell-(jMax-1)+j,0])

        left = -1
        bottom = center + 1
        right = 1 + (jMax-1-(j+1))*(iMax-1)
        top = center - 1

        if (j==0):
            top = -1
        elif (j==(jMax-2)):
            bottom = -1

        cellNeighboring.append([left, bottom, right, top])

    return (cellNeighboring)

def CellNodalNumber(X, Y, iMax, jMax):
    # Cell nodal number
    nPoints = iMax*jMax
    nGhostPoints = 2*(iMax + jMax)
    nTotalPoints = nPoints + nGhostPoints
    nInnerCell = (iMax-1)*(jMax-1)
    nGhostCell = 2*((iMax-1) + (jMax-1))
    nTotalCell = nInnerCell+nGhostCell

    cellNodalNumber = []

    # Cell Nodal Number
    # inner cell
    for j in range(0,jMax-1):
        for i in range(0,iMax-1):
            bottomLeft = i+1+j*iMax
            bottomRight = bottomLeft+1
            upperRight = bottomRight+iMax
            upperLeft = upperRight-1
            
            cellNodalNumber.append([bottomLeft, bottomRight, upperRight, upperLeft])

    # ghost cell: bottom
    for i in range(0,iMax-1):
        index = nPoints + i
        
        bottomLeft = index+1
        bottomRight = bottomLeft+1
        upperRight = bottomRight - nPoints
        upperLeft = upperRight-1

        cellNodalNumber.append([bottomLeft, bottomRight, upperRight, upperLeft])

    # ghost cell: right
    for j in range(0,jMax-1):    
        bottomLeft = iMax*(j+1)
        bottomRight = nPoints + iMax + j + 1
        upperRight = bottomRight + 1
        upperLeft = iMax*(j+2)

        cellNodalNumber.append([bottomLeft, bottomRight, upperRight, upperLeft])

    # ghost cell: top
    for i in range(0,iMax-1):
        index = nPoints + iMax + jMax + i
        
        bottomLeft = iMax*jMax - i - 1
        bottomRight = bottomLeft+1
        upperRight = index+1
        upperLeft = upperRight+1

        cellNodalNumber.append([bottomLeft, bottomRight, upperRight, upperLeft])

    # ghost cell: left
    for j in range(0,jMax-1):    
        bottomLeft = nPoints + 2*iMax + jMax + j + 2
        bottomRight = iMax*(jMax-2-j)+1  #nPoints + iMax + j + 1
        upperRight = bottomRight + iMax
        upperLeft = bottomLeft - 1

        cellNodalNumber.append([bottomLeft, bottomRight, upperRight, upperLeft])

    return (cellNodalNumber)

def CellNumber(X, Y, iMax, jMax):
    # Cell number
    nInnerCell = (iMax-1)*(jMax-1)
    nGhostCell = 2*((iMax-1) + (jMax-1))
    nTotalCell = nInnerCell+nGhostCell

    cellNumber = np.zeros(shape=(nTotalCell,1))

    # inner cell
    for j in range(0,jMax-1):
        for i in range(0,iMax-1):
            index = i+j*(iMax-1) + 1
            cellNumber[index-1,0] = index

    # ghost cell: bottom-top
    for i in range(1,iMax):
        # Bottom
        index = i + (nInnerCell-1) + 1
        cellNumber[index-1,0] = index

        # Top
        index = nTotalCell-(iMax+jMax-2) + i
        cellNumber[index-1,0] = index

    # ghost cell: left-right
    for i in range(1,jMax):
        # left
        index = nTotalCell-(jMax-2) + i - 1
        cellNumber[index-1,0] = index

        # Right
        index = nInnerCell+(iMax-2) + i + 1
        cellNumber[index-1,0] = index

    return (cellNumber)

def CellTypes(cellNodalNumber, iMax, jMax):
    nInnerCell = (iMax-1)*(jMax-1)
    nGhostCell = 2*((iMax-1) + (jMax-1))
    nTotalCell = nInnerCell+nGhostCell

    cellType = np.zeros(shape=(nTotalCell,1))

    # Cell types
    for i in range(len(cellNodalNumber)):
        cellType[i,0] = int(len(cellNodalNumber[i]))

    return (cellType)

def Initialization(chord, rOutflow, rFarfield, wakeChord, thickness, 
                    nAirfoil, nAirfoilWake, nOutflow, nWake, iMax, jMax):
    
    # Initialize
    X = np.zeros(shape=(iMax,jMax))
    Y = np.zeros(shape=(iMax,jMax))

    # Initialize coordinates
    # titik A
    X[0,0] = rOutflow
    Y[0,0] = 0

    # titik B
    X[nWake-1,0] = 0
    Y[nWake-1,0] = 0

    # titik C
    X[nWake-1 + nAirfoil-1,0] = 0
    Y[nWake-1 + nAirfoil-1,0] = 0

    # titik D
    X[iMax-1,0] = rOutflow
    Y[iMax-1,0] = 0

    # titik E
    X[iMax-1,jMax-1] = rOutflow
    Y[iMax-1,jMax-1] = rFarfield

    # titik F
    X[(iMax-1)-(nWake-1),jMax-1] = chord
    Y[(iMax-1)-(nWake-1),jMax-1] = rFarfield

    # titik G
    X[nWake-1,jMax-1] = chord
    Y[nWake-1,jMax-1] = -rFarfield

    # titik H
    X[0,jMax-1] = rOutflow
    Y[0,jMax-1] = -rFarfield

    # Initialize Boundary coordinates
    # Vertical left and right
    for j in range(1, jMax-1):
        eta = j/(jMax-1)
        m = rFarfield

        # Distribution
        A = 2   # exponential

        # Left
        X[0,j] = rOutflow
        #Y[0,j] = -m*eta     # linear
        Y[0,j] = -m*(math.exp(A*eta)-1)/(math.exp(A)-1) #exponential

        # Right
        X[iMax-1,j] = rOutflow
        #Y[iMax-1,j] = m*eta # linear
        Y[iMax-1,j] = m*(math.exp(A*eta)-1)/(math.exp(A)-1) # exponential

    # Top Boundary: Horizontal
    for i in range(1, nWake-1):
        xi = i/(nWake-1)
        m = rOutflow-1

        # Top: Bottom
        X[i,jMax-1] = rOutflow - m*xi   # linear
        #X[i,jMax-1] = rOutflow - m*math.sin(0.5*math.pi*xi)    # cosinus
        Y[i,jMax-1] = -rFarfield
        
        # Top: Top
        startIndex = iMax-1 
        X[startIndex-i,jMax-1] = rOutflow - m*xi
        Y[startIndex-i,jMax-1] = rFarfield

    # Top Boundary: C-shaped
    for i in range(1, 2*nAirfoil-1):
        m = rFarfield
        xi = i/(2*nAirfoil-1)                 # linear
        #xi = 1-math.cos(0.5*math.pi*xi)     # cosinus

        # linear distribution
        startIndex = nWake-1
        X[startIndex+i,jMax-1] = m*math.sin(-math.pi*xi) + chord
        Y[startIndex+i,jMax-1] = -m*math.cos(-math.pi*xi)

    # Bottom Boundary: Outflow
    for i in range(1,nOutflow):
        xi = i/(nOutflow-1)
        m = rOutflow-(chord+wakeChord)

        X[i,0] = rOutflow - m*xi                            # linear
        #X[i,0] = rOutflow - m*math.sin(0.5*math.pi*xi)     # cosinus
        Y[i,0] = 0

        X[(iMax-1)-i,0] = X[i,0]                                    # linear
        #X[iMax-1)-i,0] = rOutflow - m*math.sin(0.5*math.pi*xi)     # cosinus
        Y[(iMax-1)-i,0] = Y[i,0]

    # Bottom Boundary: Airfoil wake
    for i in range(1,nAirfoilWake):
        xi = i/(nAirfoilWake-1)
        m = chord + wakeChord - chord

        # Bottom: Bottom
        startIndex1 = nOutflow-1
        #X[startIndex+i,0] = chord + m*xi   # linear
        X[startIndex1+i,0] = chord + wakeChord - m*math.sin(0.5*math.pi*xi)    # cosinus
        Xrel = (X[startIndex1+i,0]-chord)/wakeChord
        Y[startIndex1+i,0] = NACA0012(Xrel, thickness, wakeChord)
        
        # Bottom: Top
        startIndex2 = (iMax-nOutflow)
        X[startIndex2-i,0] = X[startIndex1+i,0]
        Y[startIndex2-i,0] = -Y[startIndex1+i,0]

    # Bottom Boundary: Airfoil
    for i in range(1,nAirfoil):
        xi = i/(nAirfoil-1)
        m = chord

        # Bottom: Bottom
        startIndex1 = nWake-1
        #X[startIndex+i,0] = chord + m*xi   # linear
        X[startIndex1+i,0] = chord - m*math.sin(0.5*math.pi*xi)    # cosinus
        Xrel = (X[startIndex1+i,0]-0)/chord
        Y[startIndex1+i,0] = NACA0012(Xrel, thickness, chord)

        # Bottom: Top
        startIndex2 = (iMax-nWake)
        X[startIndex2-i,0] = X[startIndex1+i,0]
        Y[startIndex2-i,0] = -Y[startIndex1+i,0]

    return (X, Y)

def LaplaceSmoothing(X, Y, iMax, jMax, omega, targetError):
    # Laplace Smoothing
    Rx = np.zeros(shape=(iMax-1, jMax-1))
    Ry = np.zeros(shape=(iMax-1, jMax-1))

    iteration = 0
    error = 1
    lastRValue = 1
    residual = []

    while (error>targetError):
        for i in range(1,iMax-1):
            for j in range(1,jMax-1):
                xXi = (X[i+1,j]-X[i-1,j])/2
                yXi = (Y[i+1,j]-Y[i-1,j])/2
                xEta = (X[i,j+1]-X[i,j-1])/2
                yEta = (Y[i,j+1]-Y[i,j-1])/2
                J = xXi*yEta - xEta*yXi

                alpha = xEta**2 + yEta**2
                beta = xXi*xEta + yXi*yEta
                gamma = xXi**2 + yXi**2
                
                # Finding X
                Rx1 = alpha*(X[i+1,j] - 2*X[i,j] + X[i-1,j])
                Rx2 = (-0.5)*beta*(X[i+1,j+1] - X[i-1,j+1] - X[i+1,j-1] + X[i-1,j-1])
                Rx3 = gamma*(X[i,j+1] - 2*X[i,j] + X[i,j-1])
                Rx[i,j] = Rx1 + Rx2 + Rx3

                # Finding Y
                Ry1 = alpha*(Y[i+1,j] - 2*Y[i,j] + Y[i-1,j])
                Ry2 = (-0.5)*beta*(Y[i+1,j+1] - Y[i-1,j+1] - Y[i+1,j-1] + Y[i-1,j-1])
                Ry3 = gamma*(Y[i,j+1] - 2*Y[i,j] + Y[i,j-1])
                Ry[i,j] = Ry1 + Ry2 + Ry3

                # Update X and Y
                X[i,j] = X[i,j] + omega*((Rx[i,j])/(2*(alpha + gamma)))
                Y[i,j] = Y[i,j] + omega*((Ry[i,j])/(2*(alpha + gamma)))
        
        # Find residual
        currentRValue = np.sqrt(np.sum(Rx)**2 + np.sum(Ry)**2)
        error = abs(lastRValue - currentRValue)
        
        # Store residual
        iteration = iteration + 1
        
        # Other escape routes
        if (iteration>1000):
            break
        
        residual.append(error*100)
        
        # Update value
        lastRValue = currentRValue

    return (X, Y, residual)

def MeshQuality(X, Y, iMax, jMax):
    # Mesh Quality
    meshArea = np.zeros(shape=(iMax-1, jMax-1))
    meshSkewness = np.zeros(shape=(iMax-1, jMax-1))

    for i in range(iMax-1):
        for j in range(jMax-1):
            p = np.array([X[i,j+1] - X[i,j], Y[i,j+1] - Y[i,j]])
            q = np.array([X[i+1,j+1] - X[i,j+1], Y[i+1,j+1] - Y[i,j+1]])
            r = np.array([X[i+1,j+1] - X[i+1,j], Y[i+1,j+1] - Y[i+1,j]])
            s = np.array([X[i+1,j] - X[i,j], Y[i+1,j] - Y[i,j]])

            # Mesh Area
            area1 = -np.cross(p,s)
            area2 = np.cross(-q,-r)
            meshArea[i,j] = 0.5*(area1 + area2)

            # Skewness
            teta1 = math.degrees(math.atan2(-np.cross(p,s),np.dot(p,s)))
            teta2 = math.degrees(math.atan2(-np.cross(-s,r),np.dot(-s,r)))
            teta3 = math.degrees(math.atan2(-np.cross(-r,-q),np.dot(-r,-q)))
            teta4 = 360 - (teta1 + teta2 + teta3)
            teta = np.array([teta1, teta2, teta3, teta4])        
            tetaMaxOnMesh = (np.max(teta)-90)/90
            tetaMinOnMesh = (90-np.min(teta))/90
            skewness = max(np.array([tetaMaxOnMesh, tetaMinOnMesh]))
            meshSkewness[i,j] = skewness
    
    print("")
    print("# --------------------------------------- #")
    print("             MESH QUALITY CHECK")
    print("# --------------------------------------- #")
    print("minimum mesh area = %.2e" % np.min(meshArea))
    print("maximum mesh area = %.4f" % np.max(meshArea))
    print("ortoghonality scale. 0 = Orthogonal")
    print("minimum value = %.2e" % np.min(meshSkewness))
    print("maximum value = %.4f" % np.max(meshSkewness))
    print("")

def NodesCoordinates(X, Y, iMax, jMax):
    # Create Basic Points
    nPoints = iMax*jMax
    nGhostPoints = 2*(iMax + jMax)
    nTotalPoints = nPoints + nGhostPoints

    basicCoor = np.zeros(shape=(3, nTotalPoints))

    # Point internal mesh
    for j in range(jMax):
        for i in range(iMax):
            index = i + j*iMax

            # Store
            basicCoor[0, index] = index + 1
            basicCoor[1, index] = X[i,j]
            basicCoor[2, index] = Y[i,j]

    # Ghost Points
    for i in range(iMax):
        # Bottom
        index = i+iMax*jMax
        basicCoor[0, index] = index + 1
        basicCoor[1, index] = X[i,0]
        basicCoor[2, index] = 2*Y[i,0] - Y[i,1]

        # Top
        index = i + (iMax+1)*(jMax+1)-1
        basicCoor[0, index] = index + 1
        basicCoor[1, index] = X[iMax-1-i,jMax-1]
        basicCoor[2, index] = 2*Y[iMax-1-i,jMax-1] - Y[iMax-1-i,jMax-2]

    for j in range(jMax):
        # Right
        index = j + iMax*(jMax+1)
        basicCoor[0, index] = index + 1
        basicCoor[1, index] = 2*X[iMax-1, j] - X[iMax-2, j]
        basicCoor[2, index] = Y[iMax-1, j]

        # Left
        index = (nTotalPoints-1) - j
        basicCoor[0, index] = index + 1
        basicCoor[1, index] = 2*X[0, j] - X[1, j]
        basicCoor[2, index] = Y[0, j]
    
    return (basicCoor)

def TFI(X, Y, u, v, iMax, jMax):
    # Transfinite Interpolation
    for i in range(1,iMax-1):
        for j in range(1,jMax-1):
            U = (1-u[i,j])*X[0,j] + u[i,j]*X[iMax-1,j]
            V = (1-v[i,j])*X[i,0] + v[i,j]*X[i,jMax-1]
            UV = u[i,j]*v[i,j]*X[iMax-1,jMax-1] + u[i,j]*(1-v[i,j])*X[iMax-1,0] +\
                (1-u[i,j])*v[i,j]*X[0,jMax-1] + (1-u[i,j])*(1-v[i,j])*X[0,0]
            X[i,j] = U + V - UV

            U = (1-u[i,j])*Y[0,j] + u[i,j]*Y[iMax-1,j]
            V = (1-v[i,j])*Y[i,0] + v[i,j]*Y[i,jMax-1]
            UV = u[i,j]*v[i,j]*Y[iMax-1,jMax-1] + u[i,j]*(1-v[i,j])*Y[iMax-1,0] +\
                (1-u[i,j])*v[i,j]*Y[0,jMax-1] + (1-u[i,j])*(1-v[i,j])*Y[0,0]
            Y[i,j] = U + V - UV

    return (X, Y)

def WriteDataStructures(basicCoor, cellNeighboring, cellNodalNumber, cellType,
                        boundaryFlags, iMax, jMax):

    nPoints = iMax*jMax
    nGhostPoints = 2*(iMax + jMax)
    nTotalPoints = nPoints + nGhostPoints

    nInnerCell = (iMax-1)*(jMax-1)
    nGhostCell = 2*((iMax-1) + (jMax-1))
    nTotalCell = nInnerCell+nGhostCell
    
    # WRITE POINT COORDINATES
    f = open("nodes-coordinates", "w+")
    f.write("/*---------------------------------------------------------------------------*\\")
    f.write("\nDescription\n")
    f.write("{\n")
    f.write("\tobject\t\t\t\tnodes-coordinates;\n")
    f.write("\tnumber-of-nodes\t\t%d;\n" % nTotalPoints)
    f.write("\tinternal-nodes\t\t%d;\n" %nPoints)
    f.write("\tghost-nodes\t\t\t%d;\n" %nGhostPoints)
    f.write("}\n")
    f.write("\*---------------------------------------------------------------------------*/")
    f.write("\n\n")
    f.write("%d\n" % nTotalPoints)
    f.write("(\n")
    for i in range(nTotalPoints):
        f.write("\t(%d, " % basicCoor[0, i])
        coor = np.array([round(basicCoor[1,i],4), round(basicCoor[2,i],4)])
        f.write("%s)\n " % coor)
    f.write(")")
    f.close()

    # WRITE CELL CONNECTIVITY
    f = open("cells-connectivity", "w+")
    f.write("/*---------------------------------------------------------------------------*\\")
    f.write("\nDescription\n")
    f.write("{\n")
    f.write("\tobject\t\t\t\tcells-connectivity;\n")
    f.write("\tnumber-of-cells\t\t%d;\n" %nTotalCell)
    f.write("\tinternal-cells\t\t%d;\n" %nInnerCell)
    f.write("\tghost-cells\t\t\t%d;\n" %nGhostCell)
    f.write("}\n")
    f.write("\*---------------------------------------------------------------------------*/")
    f.write("\n\n")
    f.write("%d\n" % nTotalCell)
    f.write("(\n")
    for i in range(len(cellNeighboring)):
        f.write("\t(%d, " % (i+1))
        f.write("%s, " % cellNeighboring[i])
        f.write("%s, " % cellNodalNumber[i])
        f.write("%d)\n" % cellType[i])
    f.write(")")
    f.close()

    # WRITE BOUNDARY FLAGS
    f = open("boundary-flags", "w+")
    f.write("/*---------------------------------------------------------------------------*\\")
    f.write("\nDescription\n")
    f.write("{\n")
    f.write("\tobject\t\tboundary-flags;\n")
    f.write("}\n")
    f.write("\*---------------------------------------------------------------------------*/")
    f.write("\n\n")
    f.write("%d\n" % len(boundaryFlags))
    f.write("(\n")
    f.write("\tsolid\t\t1\t: %s\n"% boundaryFlags[0])
    f.write("\tinlet\t\t2\t: %s\n"% boundaryFlags[1])
    f.write("\toutlet\t\t3\t: %s\n"% boundaryFlags[2])
    f.write("\tfarfield\t4\t: %s\n"% boundaryFlags[3])
    f.write("\tsymmetry\t5\t: %s\n"% boundaryFlags[4])
    f.write("\tperiodic\t6\t: %s\n"% boundaryFlags[5])
    f.write(")")
    f.close()

# --------------------------------------------------------------------------------- #