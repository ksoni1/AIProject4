
# coding: utf-8

# In[2]:

###############################################################################
# Komal Soni
# CMSC-471
# Project 4 : K-means Clustering
# used an implentation of lloyd's algorithm to perform k-means clustering
##############################################################################
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
#import matplotlib.colors as colors

########################################################################
#getData(fileName): read the data points from the file and return them
########################################################################
def getData(fileName):
    Coords = []
    #open the file and start reading the data points
    with open(fileName, "r") as inputFile:
        for line in inputFile:
            #look to see where there is a break in the points
            line = [ float(i) for i in line.split() ]
            if len(line) == 2:
                Coords.append(line)
    #Coords = [[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0], [3.5, 5.0], [4.5, 5.0], [3.5, 4.5]]
    #store the array of points and return it
    Coords = np.array(Coords)
    return Coords

#################################################################################
# printClusters(centerPt): print the graph divided into their given clusters
##################################################################################
def printClusters(centerPt):
    ## iterate through the data
    for i in centerPt[1]:
        cluster = centerPt[1][i]
        for j in cluster:
            # plot all the points in the cluster and connect them with a line to its specified clusters
            plt.scatter(j[0], j[1], color= '#ababab') 
            plt.plot([centerPt[0][i][0], j[0]],  [centerPt[0][i][1], j[1] ], color='#afeeee' )
    plt.show()
    
########################################################
# cluster_points(X, mu): find the best mukey(uk) value
########################################################
def cluster_points(X, mu):
    clusters = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]]))
        for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
############################################################################
# reevaluate_centers(mu, clusters): Given a set of clusters,
#                                 the centers are recalculated as the means 
#                                  of all points belonging to a cluster.
#############################################################################
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

########################################################
# has_converged(mu, oldmu): 
########################################################
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

##########################################################
# find_centers(X, K): From given set of points, find the
#                     clusters that containd the points 
#                     closest in distance to each center.
##########################################################
def find_centers(X, K):
    oldmu = random.sample(list(X), K)
    mu = random.sample(list(X), K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        clusters = cluster_points(X, mu)
        mu = reevaluate_centers(oldmu, clusters)
    return (mu, clusters)


########################################################
# main(): check if command line arguments are valid
########################################################
def main(argv):
    if len(argv) != 3:
        print("Invalid Number of Arguments!!!")
        exit()
        
    else:
        numClusters = int(argv[1])
        
        if (numClusters <= 0):
            print("Number of Clusters must be a Positive Number!!!!!")
            exit()
        else:
            Coords = getData(argv[2])
            if (numClusters > len(Coords)):
                print("Number of Clusters must be less than or equal to Number of Points!!!!")
                exit()
            else:
                centerPt = find_centers(Coords, numClusters)
                printClusters(centerPt)

main(sys.argv)


# In[ ]:




# In[ ]:




# In[ ]:



