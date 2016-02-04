import arff
import sys

classDict = {}
conditionalCounts = {}
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
      value = row[i]
      if value not in conditionalCounts[label][i]:
        conditionalCounts[label][i][value] = 1
      else:
        conditionalCounts[label][i][value] += 1

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

if __name__ == "__main__":

  if len(sys.argv) != 4:
    print 'Error. Invalid number of arguments specified. Quitting...'
    sys.exit(-1)
  trainFileName = sys.argv[1]
  testFileName = sys.argv[2]
  if sys.argv[3] == 'n':
    trainNaiveBayes(trainFileName)
    numCorrectExamples = classifyNaiveBayes(testFileName)
    print '\n'
    print numCorrectExamples
  elif sys.argv[3] == 't':
    print 'TAN'
  else:
    print 'Error. Invalid algorithm specified. Quitting...'
    sys.exit(-1)
  
  
