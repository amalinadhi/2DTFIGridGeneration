import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------- #

def plotBarQuality(skewnessData, title):
    label = ["< 0.2", "< 0.4", "< 0.6", "< 0.8", "> 0.8"]
    index = np.arange(len(label))

    fig = plt.figure(figsize=(12, 9))
    ax = plt.subplot(111)

    ax.bar(index, skewnessData)
    plt.xlabel("Skewness Index")
    plt.ylabel("Number of cells")
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='0.65', linestyle='-')
    ax.grid(b=True, which='minor', color='0.65', linestyle='--')
    plt.xticks(index, label)
    plt.title("Cells Skewness Index")
    plt.show()

    # Save figure
    fName = "results/" + title + ".png"
    fig.savefig(fName, dpi=150)

def plotGrid(X, Y, chartTitle ='', lineColor='b', lineWidth=1, 
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
    fig = plt.figure(figsize=(12, 9))
    ax = plt.subplot(111)

    # point scatter
    if (activatePoint==True):
        ax.scatter(X, Y, c=pointColor, s=pointSize)
    else:
        pass        
    
    # line plot
    for i in range(xdiv):
        ax.plot(X[i,:], Y[i,:], lineColor, lineWidth)
    
    for j in range(ydiv):
        ax.plot(X[:,j], Y[:,j], lineColor, lineWidth)

    ax.minorticks_on()
    ax.grid(b=True, which='major', color='0.65', linestyle='-')
    ax.grid(b=True, which='minor', color='0.65', linestyle='--')
    plt.title("2D Structured Grid Generation")
    plt.show()

    # Save figure
    fName = "results/" + chartTitle + ".png"
    fig.savefig(fName, dpi=150)

def plotResidual(Residual):
    print("")
    print("# --------------------------------------- #")
    print("             SMOOTHING RESULTS")
    print("# --------------------------------------- #")
    print("number of iteration    = %d" % len(Residual))
    print("Root mean square error = {:.2%}".format(Residual[-1]))
    print("")

    index = [i for i in range(len(Residual))]

    # Generating plot
    fig = plt.figure(figsize=(12, 9))
    ax = plt.subplot(111)

    plt.plot(index[2:], Residual[2:], color='r')
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='b', linestyle='-')
    ax.grid(b=True, which='minor', color='0.65', linestyle='--')
    plt.xlabel("Iteration - ")
    plt.ylabel("RMSE [%]")
    plt.title("Smoothing Residual")
    plt.show()

    # Save figure
    fName = "results/" + "smoothing rmse history" + ".png"
    fig.savefig(fName, dpi=150)

def plotQualityComparison(data1, data2):
    label = ["< 0.2", "< 0.4", "< 0.6", "< 0.8", "> 0.8"]
    index = np.arange(len(label))

    fig = plt.figure(figsize=(12, 9))
    ax = plt.subplot(111)
    width = 0.35

    rects1 = ax.bar(index-width/2, data1, width, label="Before Smoothing")
    rects2 = ax.bar(index+width/2, data2, width, label="After Smoothing")

    plt.xlabel("Skewness Index")
    plt.ylabel("Number of cells")
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='0.65', linestyle='-')
    ax.grid(b=True, which='minor', color='0.65', linestyle='--')
    ax.legend()
    plt.xticks(index, label)
    plt.title("Cells Skewness Index")
    plt.show()

    # Save figure
    fName = "results/" + "Skewness Comparison" + ".png"
    fig.savefig(fName, dpi=150)

# ---------------------------------------------------------------------------- #