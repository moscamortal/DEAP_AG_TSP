#!/usr/bin/env python
# coding: utf-8

# In[77]:


import matplotlib.pyplot as plt

import sys
import array
import random
import numpy
import csv

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

numCities = 101

x = []
y = []

with open('100_citys.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:       
            x.append((float(row[1])/100))
            y.append((float(row[2])/100))
            line_count += 1
    x = numpy.asarray(x, dtype=numpy.int32)
    y = numpy.asarray(y, dtype=numpy.int32)

creator.create("FitnessMin",base.Fitness, weights=(-1.0,))
creator.create("Individual",array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("indices", random.sample,range(numCities),numCities)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def evalTSP(individual):
    diffx = numpy.diff(x[individual])
    diffy = numpy.diff(y[individual])
    distance = numpy.sum(diffx**2 + diffy**2)
    return distance,

toolbox.register("evaluate",evalTSP)


# In[78]:


def main():
    
    pop = toolbox.population(n=300)
    
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind:ind.fitness.values)
    stats.register("avg",numpy.mean)
    stats.register("std",numpy.std)
    stats.register("min",numpy.min)
    stats.register("max",numpy.max)
    
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 1000, stats=stats, halloffame=hof)
    
    ind = hof[0]
    plt.figure(2)
    plt.plot(x[ind],y[ind])
    plt.ion()
    plt.show()
    plt.pause(0.001)
    
    return pop, stats, hof


# In[79]:


if __name__ == "__main__":
    main()


# In[ ]:




