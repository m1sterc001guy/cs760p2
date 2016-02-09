import arff
import sys
import math
from collections import deque

classDict = {}
conditionalCounts = {}
counts = {}
k = 1

def getPartProbForLabel(label, total, row, testData):
  numOfLabel = classDict[label]
  numOfType = len(testData['attributes'][len(row) - 1][1])
  prior = float((numOfLabel + k)) / float((total + (k * numOfType)))
  returnVal = prior

  for i in xrange(0, len(row) - 1):
    value = row[i]
    numOfValue = 0
    if value in conditionalCounts[label][i]:
      numOfValue = conditionalCounts[label][i][value]
    numOfTypeValue = len(testData['attributes'][i][1])
    conditionalProb = float((numOfValue + k)) / float((numOfLabel + (k * numOfTypeValue)))
    returnVal *= conditionalProb
  return returnVal

def getLabelProb(label, row, testData):
  total = 0
  for currlabel, count in classDict.items():  
    total += count

  # compute denominator
  denominator = 0
  for currlabel, count in classDict.items():
    denominator += getPartProbForLabel(currlabel, total, row, testData)

  numerator = getPartProbForLabel(label, total, row, testData)
  return float(numerator) / float(denominator)

def printNaiveBayesStructure(testData):
  for i in xrange(0, len(testData['attributes']) - 1):
    attr = testData['attributes'][i]
    label = testData['attributes'][len(testData['attributes']) - 1][0]
    print attr[0] + ' ' + label

def trainNaiveBayes(trainFileName):
  try:
    data = arff.load(open(trainFileName, 'rb'))
  except IOError:
    print 'Error. Invalid training set file name specified. Quitting...'
    sys.exit(-1)
  for row in data['data']:
    label = row[len(row) - 1]
    if label in classDict:
      classDict[label] += 1
    else:
      classDict[label] = 1

  for row in data['data']:
    label = row[len(row) - 1]
    if label not in conditionalCounts:
      conditionalCounts[label] = {}
    for i in xrange(0, len(row) - 1):
      if i not in conditionalCounts[label]:
        conditionalCounts[label][i] = {}
      if i not in counts:
        counts[i] = {}
      value = row[i]
      if value not in conditionalCounts[label][i]:
        conditionalCounts[label][i][value] = 1
      else:
        conditionalCounts[label][i][value] += 1
      if value not in counts[i]:
        counts[i][value] = 1
      else:
        counts[i][value] += 1

def classifyNaiveBayes(testFileName):
  try:
    testData = arff.load(open(testFileName, 'rb'))
  except IOError:
    print 'Error. Invalid test set file name specified. Quitting...'
    sys.exit(-1)
  printNaiveBayesStructure(testData)
  print '\n'
  examplesClassifiedCorrectly = 0
  for row in testData['data']:

    actualLabel = row[len(row) - 1]
    maxProb = 0.0
    calcLabel = ""
    for label, count in classDict.items():
      prob = getLabelProb(label, row, testData)
      if prob > maxProb:
        maxProb = prob
        calcLabel = label
    if calcLabel == actualLabel:
      examplesClassifiedCorrectly += 1
    print calcLabel + ' ' + actualLabel + ' ' + str(maxProb)
  return examplesClassifiedCorrectly


def getJointProb(index1, value1, index2, value2, label, data):
  total = 0
  for currlabel, count in classDict.items():  
    total += count

  jointCount = 0
  for row in data['data']:
    if row[index1] == value1 and row[index2] == value2 and row[len(row) - 1] == label:
      jointCount += 1

  numOfType1 = len(data['attributes'][index1][1])
  numOfType2 = len(data['attributes'][index2][1])
  numOfLabel = len(data['attributes'][len(row) - 1][1])

  return float((jointCount + k)) / float(total + (k * numOfType1 * numOfType2 * numOfLabel))

def getJointConditionalProb(index1, value1, index2, value2, label, data):
  jointCount = 0
  for row in data['data']:
    if row[index1] == value1 and row[index2] == value2 and row[len(row) - 1] == label:
      jointCount += 1

  numOfType1 = len(data['attributes'][index1][1])
  numOfType2 = len(data['attributes'][index2][1])
  countOfLabel = classDict[label]

  return float((jointCount + k)) / float((countOfLabel + (k * numOfType1 * numOfType2)))

def getConditionalProb(index, value, label, data):
  jointCount = 0
  for row in data['data']:
    if row[index] == value and row[len(row) - 1] == label:
      jointCount += 1

  numOfType = len(data['attributes'][index][1])
  countOfLabel = classDict[label]

  return float((jointCount + k)) / float((countOfLabel + (k * numOfType)))

def mutualInformation(x, y, data):
  if x == y:
    return -1.0
  xValues = data['attributes'][x][1]
  yValues = data['attributes'][y][1]
  
  finalProb = 0
  for label in classDict:
    for xVal in xValues:
      for yVal in yValues:
        jointCond = getJointConditionalProb(x, xVal, y, yVal, label, data)
        xCond = getConditionalProb(x, xVal, label, data)
        yCond = getConditionalProb(y, yVal, label, data)
        ratio = jointCond / (xCond * yCond)
        logProb = math.log(ratio, 2)
        jointProb = getJointProb(x, xVal, y, yVal, label, data)
        finalProb += (jointProb * logProb)

  return finalProb

def calcMutualInfoForData(data):
  mutualInfo = {}
  attributes = data['attributes']
  for x in xrange(0, len(attributes) - 1):
    if x not in mutualInfo:
      mutualInfo[x] = {}
    for y in xrange(0, len(attributes) - 1):
      mutualInfo[x][y] = mutualInformation(x, y, data)

  return mutualInfo

def primsAlgo(mutualInfo):
  vertices = set()
  edges = {}

  vertices.add(0) # start with the first attribute in the file
  while len(vertices) < len(mutualInfo):
    maxWeight = 0.0
    fromVertToAdd = -1
    toVertToAdd = -1
    for fromVert in vertices:
      for toVert in mutualInfo[fromVert]:
        if toVert not in vertices:
          if mutualInfo[fromVert][toVert] > maxWeight:
            fromVertToAdd = fromVert
            toVertToAdd = toVert
            maxWeight = mutualInfo[fromVert][toVert]
          elif mutualInfo[fromVert][toVert] == maxWeight:
            if fromVert < fromVertToAdd:
              fromVertToAdd = fromVert
              toVertToAdd = toVert
              maxWeight = mutualInfo[fromVert][toVert]
            elif fromVert == fromVertToAdd:
              if toVert < toVerToAdd:
                fromVertToAdd = fromVert
                toVertToAdd = toVert
                maxWeight = mutualInfo[fromVert][toVert]
    vertices.add(toVertToAdd)
    if fromVertToAdd not in edges:
      edges[fromVertToAdd] = [toVertToAdd]
    else:
      edges[fromVertToAdd].append(toVertToAdd)
  return vertices, edges

def getParentsOfVert(vert, edges, data):
  # always add the class as a parent
  parents = [] 
  for fromVert in edges:
    toVerts = edges[fromVert]
    for toVert in toVerts:
      if vert == toVert:
        parents.append(fromVert)
  parents.append(len(data['attributes']) - 1)
  return parents

def getNewQueue(queue, attributeVals, vert):
  newQueue = deque()
  for oldVal in queue:
    for key in oldVal:
      for newVal in attributeVals:
        valDict = {}
        valDict[key] = oldVal[key]
        valDict[vert] = newVal
        newQueue.append(valDict)
  return newQueue

def constructCondProbTable(edges, data, currVert):
  parents = getParentsOfVert(currVert, edges, data)
  queue = deque()

  firstVals = data['attributes'][parents[0]][1]
  for val in firstVals:
    queue.append({parents[0]:val})

  for i in xrange(1, len(parents)):
    queue = getNewQueue(queue, data['attributes'][parents[i]][1], parents[i])

  cptTable = {}
  attrValues = data['attributes'][currVert][1]
  for attrVal in attrValues:
    if attrVal not in cptTable:
      cptTable[attrVal] = {}
    for combo in queue:
      cptKey = tuple(sorted(combo.items()))
      prob = tanGetCondProb(attrVal, currVert, combo, data)   
      cptTable[attrVal][cptKey] = prob
  return cptTable

def getAllCptTables(edges, data):
  attributes = data['attributes']
  cptTables = {}
  for i in xrange(0, len(attributes) - 1):
    cptTables[i] = constructCondProbTable(edges, data, i)

  total = len(data['data'])
  labelDict = {}
  numLabels = len(classDict)
  for label in classDict:
    prob = float((classDict[label] + k)) / float((total + (k * numLabels)))
    labelDict[label] = prob
    
  cptTables[len(attributes) - 1] = labelDict

  return cptTables

def tanGetCondProb(attrVal, attrIndex, combo, data):
  jointCount = 0
  for row in data['data']:
    if row[attrIndex] == attrVal:
      countRow = True
      for parentIndex in combo:
        if row[parentIndex] != combo[parentIndex]:
          countRow = False
      if countRow:
        jointCount += 1

  condCount = 0
  for row in data['data']:
    countRow = True
    for parentIndex in combo:
      if row[parentIndex] != combo[parentIndex]:
        countRow = False
    if countRow:
      condCount += 1

  numPossibilities = len(data['attributes'][attrIndex][1])

  prob = float((jointCount + k)) / float((condCount + (k * numPossibilities)))
  return prob 


if __name__ == "__main__":

  if len(sys.argv) != 4:
    print 'Error. Invalid number of arguments specified. Quitting...'
    sys.exit(-1)
  trainFileName = sys.argv[1]
  testFileName = sys.argv[2]
  trainNaiveBayes(trainFileName)
  if sys.argv[3] == 'n':
    numCorrectExamples = classifyNaiveBayes(testFileName)
    print '\n'
    print numCorrectExamples
  elif sys.argv[3] == 't':
    try:
      data = arff.load(open(trainFileName, 'rb'))
    except IOError:
      print 'Error. Invalid training set file name specified. Quitting...'
      sys.exit(-1)

    mutualInfo = calcMutualInfoForData(data)
    vertices, edges = primsAlgo(mutualInfo) 
    cptTables = getAllCptTables(edges, data)
    print cptTables[18]
  else:
    print 'Error. Invalid algorithm specified. Quitting...'
    sys.exit(-1)
  
  
