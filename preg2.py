
import  random, operator, matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math  
from csv import reader
from itertools import combinations



class Grafo:
    def __init__(self,name, x, y):
        self.name = name
        self.x = x
        self.y = y

    def peso(self, Grafo): 
        with open('distancias.csv', 'r') as data1:
            csv = reader(data1)
            pesos_csv = list(csv)            
            #print(pesos_csv)
        peso_n = int(pesos_csv[Grafo.name][self.name])
        #print(peso_n)
        return peso_n
    
    def __repr__(self):
        return  str(self.name)

    
    
class Fitness:
    def __init__(self, ruta):
        self.ruta = ruta
        self.peso = 0
        self.fitness= 0.0
    
    def Distancia(self):
        if self.peso ==0:
            Dist = 0
            for i in range(0, len(self.ruta)):
                fromGrafo = self.ruta[i]
                toGrafo = None
                if i + 1 < len(self.ruta):
                    toGrafo = self.ruta[i + 1]
                else:
                    toGrafo = self.ruta[0]
                    Dist += fromGrafo.peso(toGrafo)
            self.peso = Dist
  
        return self.peso
    
    def routa_Fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.Distancia())
        return self.fitness




#para las combinaciones
def crea_lista(GrafoList):
   
        #crea una combinacion de ruta sin repeticion
    ruta = random.sample(GrafoList, len(GrafoList))        
    #print(ruta)
    return ruta



def initialpoblacion(tam_pob, GrafoList):
    poblacion = []

    for i in range(0, tam_pob):
        poblacion.append(crea_lista(GrafoList))
    print(poblacion)
    return poblacion



def rankrutas(poblacion):
    fitnessResults = {}
    for i in range(0,len(poblacion)):
        fitnessResults[i] = Fitness(poblacion[i]).routa_Fitness()
    sorted_results=sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)
  
    return sorted_results





def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults





def matingPool(poblacion, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(poblacion[index])
    return matingpool





def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        

    childP2 = [item for item in parent2 if item not in childP1]
    print(startGene, endGene)

    print(parent1)
    print(parent2)

    print(childP1)
    print(childP2)
    child = childP1 + childP2

    print(child)
    return child



def breedpoblacion(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children





def mutate(individual, prob_mut):
    for swapped in range(len(individual)):
        if(random.random() < prob_mut):
            swapWith = int(random.random() * len(individual))
            
            Grafo1 = individual[swapped]
            Grafo2 = individual[swapWith]
            
            individual[swapped] = Grafo2
            individual[swapWith] = Grafo1
    return individual





def mutatepoblacion(poblacion, prob_mut):
    mutatedPop = []
    
    for ind in range(0, len(poblacion)):
        mutatedInd = mutate(poblacion[ind], prob_mut)
        mutatedPop.append(mutatedInd)
    return mutatedPop





def nextGeneration(currentGen, eliteSize, prob_mut):
    popRanked = rankrutas(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedpoblacion(matingpool, eliteSize)
    nextGeneration = mutatepoblacion(children, prob_mut)
    return nextGeneration




def Algoritmo_gen(poblacion, tam_pob, eliteSize, prob_mut, generaciones):
    pop = initialpoblacion(tam_pob, poblacion)
    progress = [1 / rankrutas(pop)[0][1]]
    #print("Distancia Inicial: " + str(progress[0]))
    
    for i in range(1, generaciones+1):
        
        pop = nextGeneration(pop, eliteSize, prob_mut)
        progress.append(1 / rankrutas(pop)[0][1])
        
        
        
    bestrutaIndex = rankrutas(pop)[0][0]
    mejor_camino = pop[bestrutaIndex]
    print("mejor ruta",mejor_camino)
    
    return mejor_camino

"""-------------------inicio--------------------"""

#se crea el grafo
GrafoList = []
#lleno la lista del gafo
for i in range(0,5):
    GrafoList.append(Grafo(name = i, x=int(random.random() * 50), y=int(random.random() * 50)))

mejor_ruta=Algoritmo_gen(poblacion=GrafoList, tam_pob=5, eliteSize=5, prob_mut=0.01, generaciones=100)

"""GRAFICA----------------------"""
x=[]
y=[]
for i in mejor_ruta:
  x.append(i.x)
  y.append(i.y)
x.append(mejor_ruta[0].x)
y.append(mejor_ruta[0].y)

plt.plot(x, y, '--o')
plt.xlabel('X')
plt.ylabel('Y')
ax=plt.gca()

bbox_props = dict(boxstyle="circle,pad=0.3", fc='C1', ec="black", lw=0.5)
for i in range(1,len(GrafoList)+1):
  ax.text(GrafoList[i-1].x, GrafoList[i-1].y, str(i), ha="center", va="center",
            size=8,
            bbox=bbox_props)
plt.tight_layout()
plt.show()


