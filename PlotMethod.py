import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------- #

def plotBarQuality(skewnessData):
    label = ["< 0.2", "< 0.4", "< 0.6", "< 0.8", "> 0.8"]
    index = np.arange(len(label))

    plt.bar(index, skewnessData)
    plt.xlabel("Skewness Index")
    plt.ylabel("Number of cells")
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.grid(b=True, which='minor', color='0.65', linestyle='--')
    plt.xticks(index, label)
    plt.title("Cells Skewness Index")
    plt.show()

def plotGrid(X, Y, lineColor='b', lineWidth=1, 
            activatePoint=True, pointColor='r', pointSize=10):
    
    """
    plot: To plot structured grid

        plot(X, Y, lineColor, lineWidth, activatePoint, pointColor, pointSize)

        INPUT:
            X (matrix)      - matrix with x-coordinates of gridpoints
            Y (matrix)      - matrix with y-coordinates of gridpoints
            lineColor       - color of mesh lines. Default blue
            lineWidth       - width of mesh lines.
            activatePoint   - to activate mesh points
            pointColor      - mesh points color
            pointSize       - size of mesh points

    """

    xdiv, ydiv = X.shape    # extracting size, X and Y should be the same

    # Generating plot
    # point scatter
    if (activatePoint==True):
        plt.scatter(X, Y, c=pointColor, s=pointSize)
    else:
        pass        
    
    # line plot
    for i in range(xdiv):
        plt.plot(X[i,:], Y[i,:], lineColor, lineWidth)
    
    for j in range(ydiv):
        plt.plot(X[:,j], Y[:,j], lineColor, lineWidth)

    plt.minorticks_on()
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.grid(b=True, which='minor', color='0.65', linestyle='--')
    plt.title("2D Structured Grid Generation")
    plt.show()

def plotResidual(Residual):
    print("")
    print("# --------------------------------------- #")
    print("             SMOOTHING RESULTS")
    print("# --------------------------------------- #")
    print("number of iteration    = %d" % len(Residual))
    print("Root mean square error = {:.2%}".format(Residual[-1]))
    print("")

    index = [i for i in range(len(Residual))]

    plt.plot(index[2:], Residual[2:], color='r')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='b', linestyle='-')
    plt.grid(b=True, which='minor', color='0.65', linestyle='--')
    plt.xlabel("Iteration - ")
    plt.ylabel("RMSE [%]")
    plt.title("Smoothing Residual")
    plt.show()

# ---------------------------------------------------------------------------- #
