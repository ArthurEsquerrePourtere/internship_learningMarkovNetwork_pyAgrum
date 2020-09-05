"""
@author: Arthur Esquerre-Pourtère
"""

import pyAgrum as gum
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from _utils.progress_bar import ProgressBar

####### GENERATING RANDOM MN #######

def createRandomMNTree(nodeNumber,minDomainSize=2,maxDomainSize=2):
    """
    Build a random markov network and return it, the returned markov network is a tree

    Examples
    --------
    >>> import markovNetworkLearning as mnl
    >>> mn=mnl.createRandomMNTree(15,2,4)

    Parameters
    ----------
    nodeNumber : int
            the number of variables of the markov network
    minDomainSize : int
            2 or above, default value : 2, each variable will have a random domain size randomly selected between minDomainSize and maxDomainSize
    maxDomainSize : int
            2 or above, and more than minDomainSize, default value : 2,  each variable will have a random domain size randomly selected between minDomainSize and maxDomainSize
        
    Returns
    -------
    pyAgrum.MarkovNet
            the resulting Markov network 
    """
    variablesDomainSize=np.random.randint(minDomainSize,maxDomainSize+1,size=nodeNumber)
    tree=gum.MarkovNet()
    if nodeNumber>0:
        tree.add(gum.LabelizedVariable("0","",int(variablesDomainSize[0])))
    for i in range(1,nodeNumber):
        otherNode=random.choice(list(tree.nodes()))
        tree.add(gum.LabelizedVariable(str(i),"",int(variablesDomainSize[i])))
        tree.addFactor({otherNode,str(i)})
    return tree

def addRandomEdgesMN(graph,newEdgesNumber):
    """
    Add edges to a markov Network by randomly merging factors

    Examples
    --------
    >>> mnl.addRandomEdgesMN(mn,3)

    Parameters
    ----------
    graph : pyAgrum.MarkovNet
            the original markov network
    newEdgesNumber : int
            the number of edges we want to add
        
    """
    nodes=list(graph.nodes())
    finalEdgesNumber=newEdgesNumber+len(graph.edges())
    if finalEdgesNumber>len(nodes)*(len(nodes)-1)/2:
        raise ValueError('The maximum number of edges in a graph is nodeNumber*(nodeNumber-1)/2, try adding less edges.')
    while len(graph.edges())<finalEdgesNumber:
        factors=graph.factors()
        nodeA=random.choice(nodes)
        otherNodeOrFactor=random.choice(nodes+factors)
        if(type(otherNodeOrFactor)==int):
            if {otherNodeOrFactor,nodeA} not in factors:
                graph.addFactor({otherNodeOrFactor,nodeA})
        else:
            newFactor=otherNodeOrFactor.copy()
            newFactor.add(nodeA)
            if newFactor not in factors:
                graph.eraseFactor(otherNodeOrFactor)
                graph.addFactor(newFactor)

def createRandomMarkovNet(nodeNumber,additionalEdge=0,minDomainSize=2,maxDomainSize=2):
    """
    Build a random markov network and return it :
        -Step 1: build a tree
        -Step 2: create and modify existing factors depending on additionalEdge
        -Step 3: generate random factor values

    Examples
    --------
    >>> import markovNetworkLearning as mnl
    >>> mn=mnl.createRandomMarkovNet(15,0.2,2,4)

    Parameters
    ----------
    nodeNumber : int
            the number of variables of the markov network
    additionalEdge : float
            default value : 0, determine how many factors will be merged
    minDomainSize : int
            2 or above, default value : 2, each variable will have a random domain size randomly selected between minDomainSize and maxDomainSize
    maxDomainSize : int
            2 or above, and more than minDomainSize, default value : 2, each variable will have a random domain size randomly selected between minDomainSize and maxDomainSize
        
    Returns
    -------
    pyAgrum.MarkovNet
            the resulting Markov network 
    """
    mn=createRandomMNTree(nodeNumber,minDomainSize,maxDomainSize)
    newEdgesNumber=int((nodeNumber-1)*additionalEdge)
    addRandomEdgesMN(mn,newEdgesNumber)
    mn.generateFactors()
    return mn

def createRandomMarkovNetUsingUndiGraph(nodeNumber,additionalEdge=0,minDomainSize=2,maxDomainSize=2):
    """
    Build a random markov network and return it :
        -Step 1: create a random graph using createRandomUndiGraph
        -Step 2: transform the undigraph to a markov network
        -Step 3: generate random factor values

    Examples
    --------
    >>> import markovNetworkLearning as mnl
    >>> mn=mnl.createRandomMarkovNetUsingUndiGraph(5,0.3,2,3)

    Parameters
    ----------
    nodeNumber : int
            the number of variables of the markov network
    additionalEdge : float
            default value : 0, determine the density of the graph, the graph will have  int((nodeNumber-1)*(1+additionalEdge)) edges
    minDomainSize : int
            2 or above, default value : 2, each variable will have a random domain size randomly selected between minDomainSize and maxDomainSize
    maxDomainSize : int
            2 or above, and more than minDomainSize, default value : 2, each variable will have a random domain size randomly selected between minDomainSize and maxDomainSize
        
    Returns
    -------
    pyAgrum.MarkovNet
            the resulting Markov network 
    """
    graph=createRandomUndiGraph(nodeNumber,additionalEdge)
    variablesDomainSize=np.random.randint(minDomainSize,maxDomainSize+1,size=nodeNumber)
    variablesList={node:gum.LabelizedVariable(str(node),"",int(variablesDomainSize[node])) for node in graph.nodes()}
    mn=undiGraphToMarkovNetwork(graph,variables=variablesList,domainSize=2)
    mn.generateFactors()
    return mn

####### GENERATING RANDOM UNDIGRAPH #######
    
def createRandomUndiGraphTree(nodeNumber):
    """
    Build a random undigraph and return it, the returned graph is a tree

    Examples
    --------
    >>> import markovNetworkLearning as mnl
    >>> g=mnl.createRandomUndiGraphTree(15)

    Parameters
    ----------
    nodeNumber : int
            the number of nodes in the graph

    Returns
    -------
    pyAgrum.UndiGraph
            the resulting undigraph
    """
    tree=gum.UndiGraph()
    if nodeNumber>0:
        tree.addNode()
    for i in range(1,nodeNumber):
        otherNode=random.choice(list(tree.nodes()))
        tree.addNode()
        tree.addEdge(otherNode,i)
    return tree

def addRandomEdgesUndiGraph(graph,newEdgesNumber):
    """
    Randomly add edges to an undigraph

    Examples
    --------
    >>> mnl.addRandomEdgesUndiGraph(g,3)

    Parameters
    ----------
    graph : pyAgrum.UndiGraph
            the original graph
    newEdgesNumber : int
            the number of edges we want to add    
    """
    nodes=list(graph.nodes())
    finalEdgesNumber=newEdgesNumber+len(graph.edges())
    if finalEdgesNumber>len(nodes)*(len(nodes)-1)/2:
        raise ValueError('The maximum number of edges in a graph is nodeNumber*(nodeNumber-1)/2, try adding less edges.')
    for i in range(newEdgesNumber):
        edges=graph.edges()
        nodeA=random.choice(nodes)
        nodeB=random.choice(nodes)
        while nodeA==nodeB or (nodeA,nodeB) in edges or (nodeB,nodeA) in edges:
            nodeA=random.choice(nodes)
            nodeB=random.choice(nodes)
        graph.addEdge(nodeA,nodeB)
    
def createRandomUndiGraph(nodeNumber,additionalEdge=0):
    """
    Build a random undigraph and return it :
        -Step 1: build a tree
        -Step 2: add edges depending on additionalEdge

    Examples
    --------
    >>> import markovNetworkLearning as mnl
    >>> g=mnl.createRandomUndiGraph(15,0.2)

    Parameters
    ----------
    nodeNumber : int
            the number of variables of the markov network
    additionalEdge : float
            default value : 0, determine the density of the graph, the graph will have  int((nodeNumber-1)*(1+additionalEdge)) edges
            
    Returns
    -------
    pyAgrum.UndiGraph
            the resulting undigraph
    """
    undiGraph=createRandomUndiGraphTree(nodeNumber)
    newEdgesNumber=int((nodeNumber-1)*additionalEdge)
    addRandomEdgesUndiGraph(undiGraph,newEdgesNumber)
    return undiGraph

####### SAMPLE FROM MN #######

def getOneSampleFromMN(mn):
    """
    Get one sample from a markov network

    Examples
    --------
    >>> sample=mnl.getOneSampleFromMN(mn)

    Parameters
    ----------
    mn : pyAgrum.MarkovNet
            the markov network

    Returns
    -------
    dict
            the sample, as a dictionary, keys are the variables names and values are the sampled values

    """
    variablesToSample=mn.names()
    sampledVariables={}
    while variablesToSample!=[]:
        sampledValue=gum.getPosterior(mn,target=variablesToSample[0],evs=sampledVariables).draw()
        sampledVariables.update({variablesToSample[0]:sampledValue})
        variablesToSample.pop(0)
    return sampledVariables
    
def bestSamplingOrder(mn):
    """
    Return the list of nodes of the markov network, from those having the most neighbors to those having the less

    Parameters
    ----------
    mn : pyAgrum.MarkovNet
            the markov network

    Returns
    -------
    list
            the resulting list
    """
    nodesNeighboursNB={node:len(mn.neighbours(node)) for node in mn.names()}
    order=sorted(nodesNeighboursNB,key=nodesNeighboursNB.__getitem__,reverse=True)
    return order
    
def getEvidence(mn,variable,candidateNodes):
    minConSet={str(node) for node in mn.minimalCondSet(variable,candidateNodes)}
    return {mn.variable(int(nodeId)).name() for nodeId in minConSet}
            
def givenEvidence(mn,variablesToSample):
    givenEvidences=dict()
    for i in range(len(variablesToSample)):
        givenEvidences[variablesToSample[i]]=getEvidence(mn,variablesToSample[i],variablesToSample[:i])
    return givenEvidences

def fastSampleFromMNRecursive(mn,samples,samplesNumber,variablesToSampleOrder,variablesEvidences,onlineComputedPotential,sampledVariables,prog):
    nextSampleIndex=len(sampledVariables)
    if nextSampleIndex==len(variablesToSampleOrder):
        samples.extend(sampledVariables for _ in range(samplesNumber))
        if prog!=None:
            prog.increment_amount(samplesNumber)
            prog.display()
    else:
        variable=variablesToSampleOrder[nextSampleIndex]
        variableSample=defaultdict(lambda: 0)
        sampledVariablesTuple = tuple(sampledVariables[key] for key in variablesToSampleOrder[:nextSampleIndex] if key in variablesEvidences[variable])
        if sampledVariablesTuple in onlineComputedPotential[variable]:
            posterior=onlineComputedPotential[variable][sampledVariablesTuple]
        else:
            posterior=gum.getPosterior(mn,target=variable,evs=sampledVariables)
            onlineComputedPotential[variable][sampledVariablesTuple]=posterior
        for _ in range(samplesNumber):
            value=posterior.draw()
            variableSample[value]+=1
        for value in variableSample.keys():
            fastSampleFromMNRecursive(mn,samples,variableSample[value],variablesToSampleOrder,variablesEvidences,onlineComputedPotential,{**sampledVariables,**{variable:value}},prog)
        
def fastSampleFromMN(mn,fileName,samplesNumber,shuffle=True,display=False):
    """
    Get several samples from a markov network and save it in a CSV file

    Examples
    --------
    >>> mnl.sampleFromMN(mn,"./samples/sampleMN.csv",100,display=True)

    Parameters
    ----------
    mn : pyAgrum.MarkovNet
            the markov network
    fileName : str
            the path to save the samples
    samplesNumber : int
            the number of samples
    shuffle : bool
            default value : True, True to shuffle the samples, False otherwise, shuffle is highly recommended most of the time
    display : bool
            default value : False, True to display a progress bar, False otherwise
    """
    if display:
        prog = ProgressBar(fileName + ' : ', 0, samplesNumber, 60, mode='static', char='#')
        prog.display()
    else:
        prog=None
    df=pd.DataFrame(columns=sorted(mn.names()))
    variablesToSampleOrder=bestSamplingOrder(mn)
    variablesEvidences=givenEvidence(mn,variablesToSampleOrder)
    onlineComputedPotential=dict()
    for variableName in variablesEvidences.keys():
        onlineComputedPotential[variableName]=dict()
    sampledVariables={}
    samples=[]
    fastSampleFromMNRecursive(mn,samples,samplesNumber,variablesToSampleOrder,variablesEvidences,onlineComputedPotential,sampledVariables,prog)
    if shuffle:
        random.shuffle(samples)
    df=pd.DataFrame(samples)  
    df.to_csv(fileName, index=False)
    if display:
        print("\nDone")

####### PSEUDO LOG LIKELIHOOD #######

def factorsName(mn):
    """
    Return the list of factor of a markov network, each factor is a set of node names

    Parameters
    ----------
    mn : pyAgrum.MarkovNet
            the markov network
            
    Returns
    -------
    list
            a list of set, each set is a factor and contains node names
    """
    factors=mn.factors()
    return [{mn.variable(variable).name() for variable in factor} for factor in factors]

def multiplyFactor(myList,instance,mn) :
    result = 1
    for x in myList: 
        result = result * mn.factor(x)[instance]
    return result  

def computePseudoLogLikelihoodData(mn,df):
    """
    Compute the pseudo loglikelihood, on a given markov network, of an instantiation

    Parameters
    ----------
    mn : pyAgrum.MarkovNet
            the markov network
    df : pandas.core.series.Series
            the instantiation
            
    Returns
    -------
    float
            the pseudo Loglikelihood
    """
    data=df.to_dict()
    I=gum.Instantiation()
    for i in data.keys():
        I.add(mn.variableFromName(i))
    I.fromdict(data)
    sumPseudoLogLikelihood=0
    factors=factorsName(mn)
    
    for variable in data.keys():
        originalValue=I[variable]
        variable=mn.variableFromName(variable)
        variableFactors=[factor for factor in factors if variable.name() in factor]
        phat=multiplyFactor(variableFactors,I,mn)
        sommephat=sum([multiplyFactor(variableFactors,
                                      I.chgVal(variable,int(value)),mn) for value in range(variable.domainSize())])
        I.chgVal(variable,originalValue)
        sumPseudoLogLikelihood+=np.log(phat/sommephat)
    return sumPseudoLogLikelihood


def computePseudoLogLikelihoodDataset(mn,df):
    """
    Compute the pseudo loglikelihood, on a given markov network, of a dataset

    Parameters
    ----------
    mn : pyAgrum.MarkovNet
            the markov network
    df : pandas.core.frame.DataFrame
            the dataset
            
    Returns
    -------
    float
            the pseudo Loglikelihood
    """
    return df.apply(lambda data : computePseudoLogLikelihoodData(mn,data), axis = 1).sum()


####### PARAMETERS LEARNING #######
    
def bestDepart(mn,df):
    """
    Replace the factors values of a markov network by their joint probability in a given dataset

    Parameters
    ----------
    mn : pyAgrum.MarkovNet
            the markov network
    df : pandas.core.frame.DataFrame
            the dataset     
    """
    for factor in factorsName(mn):
        factorList=list(factor)
        factorList.sort()
        jointProbability=(df.groupby(factorList).size()/df.shape[0])
        for factorI in mn.factor(factor).loopIn():
            I=[factorI[value] for value in factorList]
            try:
                factorValue=jointProbability.loc[tuple(I)]
            except :
                factorValue=0.000001
            mn.factor(factor)[factorI]=factorValue
    
    
def seriesToInstantiation(mn,df):
    d=df.to_dict()
    I=gum.Instantiation()
    for i in d.keys():
        I.add(mn.variableFromName(i))
    I.fromdict(d)
    return I

def multiplyFactorDict(myList,instance,mn):
    result = 1
    for x in myList: 
        result = result * mn[tuple(sorted(x))][instance]
    return result  

def computeGradientIDataVariable(mn,mnDict,Idata,Ifactor,dictFactor,variable,variableFactors):
    """
    Compute the gradient of a value of a factor, on a variable in an instantiation
    
    Parameters
    ----------
    mn : pyAgrum.MarkovNet
            the original markov network
    mnDict : dict
            the markov network we are using to a learn, as a dictionary
    Idata : pyAgrum.Instantiation
            the instantiation
    Ifactor : pyAgrum.Instantiation
            Instantiation that describe the value of the factor
    dictFactor : dict
            Dict that describe the value of the factor
    variable : pyAgrum.DiscreteVariable
            the variable
    variableFactors : dict
            keys are variable, values are list of factors that contains the variable in their scope
            
    Returns
    -------
    float
            the gradient
    """
    gradient=0
    variableName=variable.name()
    dictFactorKeys=tuple(sorted(dictFactor.keys()))
    
    originalValue=Idata[variableName]
    variableFactors[variableName].remove(dictFactorKeys)
    Idata.chgVal(variable,Ifactor[variableName])
    up=multiplyFactorDict(variableFactors[variableName],Idata,mnDict)
    Idata.chgVal(variable,originalValue)
    variableFactors[variableName].append(dictFactorKeys)
    down=sum([multiplyFactorDict(variableFactors[variableName],
                         Idata.chgVal(variable,int(value)),mnDict) for value in range(variable.domainSize())])
    gradient-=up/down
    Idata.chgVal(variable,originalValue)
    if(Ifactor[variableName]==Idata[variableName]):
        gradient+=1/mnDict[dictFactorKeys][Ifactor]
    return gradient

def computeGradientIData(mn,mnDict,Idata,Ifactor,dictFactor,variableFactors):
    """
    Compute the gradient of a value of a factor, on an instantiation
    
    Parameters
    ----------
    mn : pyAgrum.MarkovNet
            the original markov network
    mnDict : dict
            the markov network we are using to a learn, as a dictionary
    Idata : pyAgrum.Instantiation
            the instantiation
    Ifactor : pyAgrum.Instantiation
            Instantiation that describe the value of the factor
    dictFactor : dict
            Dict that describe the value of the factor
    variableFactors : dict
            keys are variable, values are list of factors that contains the variable in their scope
            
    Returns
    -------
    float
            the gradient
    """
    keys={Idata.variable(i).name() for i in range(Idata.nbrDim())}
    gradient=0
    for i in range(Ifactor.nbrDim()):
        variable=Ifactor.variable(i)
        keys.remove(variable.name())
        if all([not(Ifactor.contains(mn.variableFromName(variable2))) or Ifactor[variable2]==Idata[variable2] for variable2 in keys]):
                
            gradient+=computeGradientIDataVariable(mn,mnDict,Idata,Ifactor,dictFactor,variable,variableFactors)
                
        keys.add(variable.name())
    return gradient
    
def computeGradientIDataset(mn,mnDict,df_I,Ifactor,dictFactor,variableFactors):
    """
    Compute the gradient of a value of a factor, on a dataset
    
    Parameters
    ----------
    mn : pyAgrum.MarkovNet
            the original markov network
    mnDict : dict
            the markov network we are using to a learn, as a dictionary
    df_I : pandas.core.frame.Series
            the dataset, contains Instantiation
    Ifactor : pyAgrum.Instantiation
            Instantiation that describe the value of the factor
    dictFactor : dict
            Dict that describe the value of the factor
    variableFactors : dict
            keys are variable, values are list of factors that contains the variable in their scope
            
    Returns
    -------
    float
            the gradient
    """
    return df_I.apply(lambda data : computeGradientIData(mn,mnDict,data,Ifactor,dictFactor,variableFactors)).sum() 

def updateMNAndComputeGradient(mn,mnDict,df_I,step,newMnDict,variableFactors,factorsName):
    """
    Perform one iteration of the gradient algorithm

    Parameters
    ----------
    mn : pyAgrum.MarkovNet
            the markov network
    mnDict : dict
            the markov network we are using to learn, as a dictionary
    df_I : pandas.core.frame.Series
            the dataset, contains Instantiation
    step : float
            the learning rate
    newMnDict : dict
            temporary markov network we are using to learn, as a dictionary
    variableFactors : dict
            keys are variable, values are list of factors that contains the variable in their scope
    factorsName : list
            the list of factor of the markov network, each factor is a set of node names
            
    Returns
    -------
    float
            the sum of the gradient
    """
    sumGradient=0
    for factor in factorsName:
        for Ifactor in mnDict[factor].loopIn():
            dictFactor=Ifactor.todict()
            gradient=computeGradientIDataset(mn,mnDict,df_I,Ifactor,dictFactor,variableFactors)
            sumGradient+=np.abs(gradient)
            if gradient > 0:
                newMnDict[factor][dictFactor]=max(mnDict[factor][Ifactor]*(1+step),0.000001)
            elif gradient < 0:
                newMnDict[factor][dictFactor]=max(mnDict[factor][Ifactor]/(1+step),0.000001)
            
    for factor in factorsName:
        mnDict[factor].fillWith(newMnDict[factor])
    return mnDict,sumGradient

def sortedFactors(factors):
    return {tuple(sorted(factor)):factor for factor in factors}

def mnToDictOfPotential(mn,factorsNames):
    mnDict=dict()
    for factor in factorsNames:
       mnDict[factor]=mn.factor(factorsNames[factor])
    return mnDict

def fillMnWithDictOfPotentiel(mn,factorsNames,mnDict):
    for factor in factorsNames:
        mn.factor(factorsNames[factor]).fillWith(mnDict[factor])
        
def copyDictOfPotential(mnDict):
    newMnDict=dict()
    for factor in mnDict:
        newMnDict[factor]=gum.Potential(mnDict[factor])
    return newMnDict

def learnParameters(mn,df,threshold=-1,maxIt=float('inf'),step=0.1,stepDiscount=1,display=False):
    """
    Learn the parameters of a markov network, using gradient algorithm, with a given the database
    The objective function we want to maximise is the pseudo-log-likelihood
    It is necessery to set a value to either threshold or maxIt

    Examples
    --------
    >>> mnl.learnParameters(forgottenMn,df,maxIt=25,step=0.15,stepDiscount=0.9,display=True)

    Parameters
    ----------
    mn : pyAgrum.MarkovNet
            the markov network
    df : pandas.core.frame.DataFrame
            the dataset
    threshold : float
            default value : -1, the algorithm stops if the sum of the gradient is less than this threshold
    maxIt : int
            default value : inf, the algorithm stops after maxIt iteration of the gradient
    step : float
            default value : 0.1, the learning rate
    stepDiscount : 1
            default value : 1, the discount of the learning rate
    display : bool
            default value : False, True to display information at each iteration, False otherwise    
    """
    #bestDepart(mn,df)
    assert maxIt>0, 'maxIt be have a strictly positive value'
    assert threshold>0 or maxIt!=float('inf'), 'you must give a strictly positive value to threshold or maxIt, otherwise the algorithm can not stop'
        
    factorsNames=sortedFactors(factorsName(mn))
    mnDict=mnToDictOfPotential(mn,factorsNames)
    for factor in factorsNames:
        for Ifactor in mnDict[factor].loopIn():
            mnDict[factor][Ifactor]=max(mnDict[factor][Ifactor],0.001)
    df_I=df.apply(lambda data : seriesToInstantiation(mn,data), axis = 1)
    newMnDict=copyDictOfPotential(mnDict)
    variableFactors=dict()
    for variable in mn.names():
        variableFactors[variable]=[factor for factor in factorsNames if variable in factor]
    mnDict,sumGradient=updateMNAndComputeGradient(mn,mnDict,df_I,step,newMnDict,variableFactors,factorsNames)
    oldGradient=sumGradient
    if display:
        print("Sum Gradient ",sumGradient)
        print("Step size : ",step)
    it=1
    while sumGradient>threshold and it<maxIt:
        mnDict,sumGradient=updateMNAndComputeGradient(mn,mnDict,df_I,step,newMnDict,variableFactors,factorsNames)
        if display:
            print("Sum Gradient ",sumGradient)
            print("Step size : ",step)
        
        if oldGradient <= sumGradient:
            step=step*stepDiscount
        oldGradient=sumGradient
        it+=1
    fillMnWithDictOfPotentiel(mn,factorsNames,mnDict)

####### UNDIGRAPH TO MARKOV NETWORK #######

def degeneracyOrdering(graph):
    """
    Return the degeneracy ordering of a graph

    Parameters
    ----------
    graph : dict
            the graph, as an adjacency list
            
    Returns
    -------
    list
            the degeneracy ordering
    """
    L = []
    d=dict()
    maxDegree=-1
    for v in graph:
        degree=len(graph[v])
        d[v]=degree
        if degree>maxDegree:
            maxDegree = degree
    D =[[] for _ in range(maxDegree+1)]
    for v in graph:
        D[len(graph[v])].append(v)
    k=0
    for _ in range(len(graph)):
        for i in range(0,maxDegree+1):
            if len(D[i]) != 0:
                break
        k=max(k,i)
        v=D[i].pop()
        L.insert(0,v)
        for w in graph[v]:
            if w not in L:
                degree = d[w]
                D[degree].remove(w)
                D[degree-1].append(w)
                d[w]-=1
    return L   

def BronKerbosch(graph,P,R,X):
    cliques=[]
    if len(P)==0 and len(X)==0:
        cliques.append(R)
    else:
        for v in list(P):
            cliques.extend(BronKerbosch(graph,P.intersection(graph[v]),R.union([v]), X.intersection(graph[v])))
            P.remove(v)
            X.add(v)
    return cliques

def removeSelfLoop(graph):
    for node in graph:
        if node in graph[node]:
            graph[node].remove(node)

def getAllMaximalCliquesDict(graph):
    """
    Return every maximal cliques of the given graph, using Bron Kerbosch algorithm

    Parameters
    ----------
    graph : dict
            the graph, as an adjacency list
            
    Returns
    -------
    list
            the list of maximal cliques
    """
    removeSelfLoop(graph)
    P = set(graph.keys())
    X = set()
    cliques = []
    order=degeneracyOrdering(graph)
    for i in range(len(order)):
        v=order[i]
        neighbors=set(graph[v])
        P=neighbors.intersection(set(order[i:]))
        X=neighbors.intersection(set(order[:i]))
        cliques.extend(BronKerbosch(graph,P,{v},X))
    return cliques

def undiGraphToDict(graph):
    """
    Return an adjacency list representing the given graph

    Parameters
    ----------
    graph : pyAgrum.UndiGraph
            the given graph
            
    Returns
    -------
    list
            the adjacency list
    """
    graphDict=dict()
    for node in graph.nodes():
        graphDict[node]=graph.neighbours(node)
    return graphDict

def getAllMaximalCliquesUndiGraph(graph):
    """
    Return every maximal cliques of the given graph, using Bron Kerbosch algorithm

    Parameters
    ----------
    graph : pyAgrum.UndiGraph
            the graph
            
    Returns
    -------
    list
            the list of maximal cliques
    """
    graphDict=undiGraphToDict(graph)
    return getAllMaximalCliquesDict(graphDict)

def undiGraphToMarkovNetwork(graph,variables=dict(),domainSize=2):
    """
    Build a markov network from an UndiGraph
    Variables can be either given in parameters or can be created if they are not defined

    Parameters
    ----------
    graph : pyAgrum.UndiGraph
            the graph
    variables : dict<str:pyAgrum.DiscreteVariable>
            already defined variables, keys are variables name
    domainSize : int
            the domain of variables that are not defined
            
    Returns
    -------
    pyAgrum.MarkovNet
            the resulting Markov network
    """
    nodes=graph.nodes()
    cliques=getAllMaximalCliquesUndiGraph(graph)
    definedVariables=variables.keys()
    mn=gum.MarkovNet()
    for node in nodes:
        if node in definedVariables:
            mn.add(variables[node])
        else:
            mn.add(gum.LabelizedVariable(str(node),"",domainSize))
    for clique in cliques:
        mn.addFactor(clique)
    return mn

def dictToMarkovNetwork(graph,variables=dict(),domainSize=2):
    """
    Build a markov network from an adjacency list
    Variables can be either given in parameters or can be created if they are not defined

    Parameters
    ----------
    graph : dict
            the graph, as an adjacency list
    variables : dict<str:pyAgrum.DiscreteVariable>
            already defined variables, keys are variables name
    domainSize : int
            the domain of variables that are not defined
            
    Returns
    -------
    pyAgrum.MarkovNet
            the resulting Markov network
    """
    for var1 in graph.keys():
        for var2 in graph[var1]:
            if var1 not in graph[var2]:
                raise ValueError('Can only convert an adjacency list into a markov network : as "'
                                 +str(var2)+'" is a neighbor of "'+str(var1)+'", then "'+str(var1)+'" must be a neighbor of "'+str(var2)+'".')
    nodes=graph.keys()
    cliques=getAllMaximalCliquesDict(graph)
    definedVariables=variables.keys()
    mn=gum.MarkovNet()
    for node in nodes:
        if node in definedVariables:
            mn.add(variables[node])
        else:
            mn.add(gum.LabelizedVariable(str(node),"",domainSize))
    for clique in cliques:
        mn.addFactor(clique)
    return mn

####### STRUCTURE LEARNING #######
    
def isIndependant(variable,Y,MB,learner,threshold):
    """
    Check whether or not two variables are independant given a set of other variables
    Use a statistical test Chi2 on a database

    True if independant, False otherwise
    """
    stat,pvalue=learner.chi2(variable,Y,list(MB))
    #stat,pvalue=learner.G2(variable,Y,list(MB))  
    return pvalue>=threshold
    
def GS(variable,V,learner,threshold):
    MB=set()
    #grow phase
    toTest=set(V)
    toTest.remove(variable)
    tested=set()
    while toTest:
        Y=toTest.pop()
        if(not isIndependant(variable,Y,MB,learner,threshold)):
            MB.add(Y)
            toTest=toTest.union(tested)
            tested.clear()
        else:
            tested.add(Y)
    #Shrink phase
    toTest=set(MB)
    tested=set()
    while toTest:
        Y=toTest.pop()
        MB.remove(Y)
        if(not isIndependant(variable,Y,MB,learner,threshold)):
            MB.add(Y)
            tested.add(Y)
        else:
            toTest=toTest.union(tested)
            tested.clear()
    return MB

def correctError(MB):
    for var1 in MB.keys():
        for var2 in MB[var1]:
            if var1 not in MB[var2]:
                MB[var2].add(var1)  

def GSMN(mn,fileName,threshold=0.05):
    """
    Learn the structure of a markov network, using GSMN algorithm on a given the database

    Examples
    --------
    >>> mn=mnl.GSMN(template,"./samples/sampleMN.csv",0.0001)

    Parameters
    ----------
    mn : pyAgrum.MarkovNet
            the template of the markov network
    fileName : str
            the other markov network
    threshold : float
            default value : 0.05, hyperparameter used for the statistical test
            
    Returns
    -------
    pyAgrum.MarkovNet
            the learned markov network
    """
    V=mn.names()
    mnVariables=dict()
    for name in V:
        mnVariables[name]=mn.variableFromName(name)
    MB=dict()
    learner=gum.BNLearner(fileName)
    for variable in V:
        MB[variable]=GS(variable,V,learner,threshold)
    correctError(MB)
    mn=dictToMarkovNetwork(MB,mnVariables)
    return mn

def IGSMNStar(X,Y,S,F,T,learner,threshold):
    if Y in T:
        return False
    if Y in F:
        return True
    return isIndependant(X,Y,S,learner,threshold)

def GSMNStar(mn,fileName,threshold=0.05):
    """
    Learn the structure of a markov network, using GSMNStar algorithm on a given the database

    Examples
    --------
    >>> mn=mnl.GSMNStar(template,"./samples/sampleMN.csv",0.0001)

    Parameters
    ----------
    mn : pyAgrum.MarkovNet
            the template of the markov network
    fileName : str
            the other markov network
    threshold : float
            default value : 0.05, hyperparameter used for the statistical test
            
    Returns
    -------
    pyAgrum.MarkovNet
            the learned markov network
    """
    V=mn.names()
    mnVariables=dict()
    for name in V:
        mnVariables[name]=mn.variableFromName(name)
    MB={X:{} for X in V}
    P=dict()
    learner=gum.BNLearner(fileName)
    for X in V:
        for Y in V:
            if(X!=Y):
                P[(X,Y)]=max(learner.chi2(X,Y)[1],0.00000001)
    avgLog={X:np.mean([np.log(P[(X,Y)]) for Y in V if X!=Y]) for X in V}
    piOrder=sorted(avgLog,key=avgLog.__getitem__)
    lambdaOrder=dict()
    for X in V:
        XPvalues={Y:P[(X,Y)] for Y in V if X!=Y}
        lambdaOrder[X]=sorted(XPvalues,key=XPvalues.__getitem__)
    examined=set()
    while piOrder!=[]:
        X=piOrder.pop(0)
        #propagation phase
        T={Y for Y in examined if X in MB[Y]}
        F={Y for Y in examined if X not in MB[Y]}
        #T={}
        #F={}
        for Y in T:
            lambdaOrder[X].append(lambdaOrder[X].pop(lambdaOrder[X].index(Y)))
        for Y in F:
            lambdaOrder[X].append(lambdaOrder[X].pop(lambdaOrder[X].index(Y)))
        #grow phase
        S=[]
        while lambdaOrder[X]!=[]:
            Y=lambdaOrder[X].pop(0)
            if P[(X,Y)]<=threshold:
                if not IGSMNStar(X,Y,S,F,T,learner,threshold):
                    S.append(Y)
                    if lambdaOrder[Y]!=[]:
                        lambdaOrder[Y].insert(0,lambdaOrder[Y].pop(lambdaOrder[Y].index(X)))
                    for i in range(len(S)-2,-1,-1):
                        W=S[i]
                        if lambdaOrder[Y]!=[]:
                            lambdaOrder[Y].insert(0,lambdaOrder[Y].pop(lambdaOrder[Y].index(W)))
        for i in range(len(S)-1,-1,-1):
            W=S[i]
            if W in piOrder:
                 piOrder.insert(0,piOrder.pop(piOrder.index(W)))     
                 break
        #shrink phase
        Scopy=S.copy()
        Scopy.reverse()
        for Y in Scopy:
            S.remove(Y)
            if not IGSMNStar(X,Y,S,F,T,learner,threshold):
                S.append(Y)
        MB[X]=set(S)
        examined.add(X)
    mn=dictToMarkovNetwork(MB,mnVariables)
    return mn

def IGSIMN(V,X,Y,S,F,T,learner,threshold,KI,KD):
    V.remove(X)
    V.remove(Y)
    S=set(S)
    if Y in T:
        return False
    if Y in F:
        return True
    for test in KD[(X,Y)]:
        if  S.issubset(test):
            return False
    
    for W in V:
        for test1 in KD[(X,W)]:
            if S.issubset(test1):
                for test2 in KD[(W,Y)]:
                    if S.issubset(test2):
                        KD[(X,Y)].append(test1.intersection(test2))
                        KD[(Y,X)].append(test1.intersection(test2))
                        return False
    for test in KI[(X,Y)]:
        if test.issubset(S):
            return True
    for W in V:
        for test1 in KI[(X,W)]:
            if test1.issubset(S):
                for test2 in KD[(W,Y)]:
                    if test1.issubset(test2):
                        KI[(X,Y)].append(test1)
                        KI[(Y,X)].append(test1)
                        return True
    t=isIndependant(X,Y,S,learner,threshold)
    if t:
        KI[(X,Y)].append(S)
        KI[(Y,X)].append(S)
    else:
        KD[(X,Y)].append(S)
        KD[(Y,X)].append(S)
    return t

def GSIMN(mn,fileName,threshold=0.05):
    """
    Learn the structure of a markov network, using GSIMN algorithm on a given the database
    THIS ALGORITHM HAS NOT BEEN TESTED AND PROBABLY DOES NOT WORK, DO NOT USE WITHOUT TESTING IT
    
    Examples
    --------
    >>> mn=mnl.GSIMN(template,"./samples/sampleMN.csv",0.0001)
    
    Parameters
    ----------
    mn : pyAgrum.MarkovNet
            the template of the markov network
    fileName : str
            the other markov network
    threshold : float
            default value : 0.05, hyperparameter used for the statistical test
            
    Returns
    -------
    pyAgrum.MarkovNet
            the learned markov network
    """
    V=mn.names()
    mnVariables=dict()
    for name in V:
        mnVariables[name]=mn.variableFromName(name)
    MB={X:{} for X in V}
    P=dict()
    KI=dict()
    KD=dict()
    learner=gum.BNLearner(fileName)
    for X in V:
        for Y in V:
            if(X!=Y):
                P[(X,Y)]=max(learner.chi2(X,Y)[1],0.0000001)             
                KI[(X,Y)]=[]
                KD[(X,Y)]=[]
                
    avgLog={X:np.mean([np.log(P[(X,Y)]) for Y in V if X!=Y]) for X in V}
    piOrder=sorted(avgLog,key=avgLog.__getitem__)
    lambdaOrder=dict()
    for X in V:
        XPvalues={Y:P[(X,Y)] for Y in V if X!=Y}
        lambdaOrder[X]=sorted(XPvalues,key=XPvalues.__getitem__)
    
    examined=set()
    while piOrder!=[]:
        X=piOrder.pop(0)
        #propagation phase
        T={Y for Y in examined if X in MB[Y]}
        F={Y for Y in examined if X not in MB[Y]}
        for Y in T:
            lambdaOrder[X].append(lambdaOrder[X].pop(lambdaOrder[X].index(Y)))
        for Y in F:
            lambdaOrder[X].append(lambdaOrder[X].pop(lambdaOrder[X].index(Y)))
        #grow phase
        S=[]
        while lambdaOrder[X]!=[]:
            Y=lambdaOrder[X].pop(0)
            if P[(X,Y)]<=threshold:
                if not IGSIMN(V,X,Y,S,F,T,learner,threshold,KI,KD):
                    S.append(Y)
                    if lambdaOrder[Y]!=[]:
                        lambdaOrder[Y].insert(0,lambdaOrder[Y].pop(lambdaOrder[Y].index(X)))
                    for i in range(len(S)-2,-1,-1):
                        W=S[i]
                        if lambdaOrder[Y]!=[]:
                            lambdaOrder[Y].insert(0,lambdaOrder[Y].pop(lambdaOrder[Y].index(W)))
                V.append(X)
                V.append(Y)
        for i in range(len(S)-1,-1,-1):
            W=S[i]
            if W in piOrder:
                 piOrder.insert(0,piOrder.pop(piOrder.index(W)))     
                 break
        #shrink phase
        Scopy=S.copy()
        Scopy.reverse()
        for Y in Scopy:
            S.remove(Y)
            if not IGSIMN(V,X,Y,S,F,T,learner,threshold,KI,KD):
                S.append(Y)
            V.append(X)
            V.append(Y)
        MB[X]=set(S)
        examined.add(X)
    mn=dictToMarkovNetwork(MB,mnVariables)
    return mn

####### PERFORMANCE TESTS #######
    
def learnMN(template,fileName,structureLearningAlgorithm,thresholdStatisticalTest=0.0001,threshold=-1,maxIt=float('inf'),step=0.05,stepDiscount=1,display=False):
    df=pd.read_csv(fileName, dtype=str)
    if display:
        print("Starting structure learning")
    learnedStructureMn=structureLearningAlgorithm(template,fileName,thresholdStatisticalTest)
    if display:
        print("Structure learned")
        print("Starting parameters learning")
    learnedStructureMn.generateFactors()
    learnParameters(learnedStructureMn,df,threshold=threshold,maxIt=maxIt,step=step,stepDiscount=stepDiscount,display=display)
    if display:
        print("Parameters learned")
    return learnedStructureMn
    
def compareMNStructure(mn1,mn2):
    """
    Compare mn1 structure with mn2 structure, mn1 is the reference

    Parameters
    ----------
    mn1 : pyAgrum.MarkovNet
            the reference markov network
    mn2 : pyAgrum.MarkovNet
            the other markov network
            
    Returns
    -------
    int
            the number of false positives
    int
            the number of false negatives
    float
            the NHD
    """
    edges1=mn1.edges()
    edges2=mn2.edges()
    nodes=mn1.nodes()
    falsePositives=edges2.difference(edges1)
    falseNegatives=edges1.difference(edges2)
    NHD=(len(falsePositives)+len(falseNegatives))/((len(nodes)*(len(nodes)-1))/2)
    return falsePositives,falseNegatives,NHD

def computePartitionFunction(mn):
    """
    Compute the partition function of a markov network

    Parameters
    ----------
    mn : pyAgrum.MarkovNet
            the markov network
            
    Returns
    -------
    float
            the partition function
    """
    P=getUnnormalizedProbabilityDistribution(mn)
    return P.sum()

def getUnnormalizedProbabilityDistribution(mn):
    P=1
    factors=[mn.factor(factor) for factor in mn.factors()]
    for factor in factors:
        P*=factor
    return P

def getNormalizedProbabilityDistribution(mn):
    P=getUnnormalizedProbabilityDistribution(mn)
    return P/P.sum()

def compareMNParametersKLDistance(mn1,mn2):
    P1=getNormalizedProbabilityDistribution(mn1)
    P2tmp=getNormalizedProbabilityDistribution(mn2)
    P2=gum.Potential(P1)
    P2.fillWith(P2tmp)
    return P1.KL(P2)

def compareMNParametersMaxProbabilityDistance(mn1,mn2):
    P1=getNormalizedProbabilityDistribution(mn1)
    P2tmp=getNormalizedProbabilityDistribution(mn2)
    P2=gum.Potential(P1)
    P2.fillWith(P2tmp)
    return (P1-P2).abs().max()
