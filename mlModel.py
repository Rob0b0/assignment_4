
import numpy as np
from sklearn import tree
import operator
from sklearn.svm import SVC
import numpy
import scipy.io as scio

def doPCA(matrix, PCnum = 5):

	matrix = np.matrix(matrix)
	covMatrix = np.cov(m = matrix.T)
	(eigenvalues, eigenvectors) = np.linalg.eig(covMatrix)
	chosenEigenvalues = np.array(eigenvalues[0:PCnum])
	chosenEigenvectors = np.matrix(eigenvectors[0:PCnum])
	newMatrix = np.dot(matrix, chosenEigenvectors.T)
	matrixReconstruct = np.dot(newMatrix, chosenEigenvectors)
	matrixReconstruct = matrixReconstruct.tolist()

	return matrixReconstruct

class DecisionTreeModel:

	def __init__(self):
		
		self.isTrained = False
		self.model = tree.DecisionTreeClassifier()

	def train(self, attributeMatrix, labelVector):
		
		self.model = self.model.fit(attributeMatrix, labelVector)
		self.isTrained = True

	def predict(self, attributeVector):

		return self.model.predict(attributeVector)

	def predict_batch(self, attributeMatrix):

		return [self.predict(attributeVector = tempAttributeVector) for tempAttributeVector in attributeMatrix]


# =================================================================================================================================


class KnnModel:

	def __init__(self):

		self.isTrained = False
		self.k = 2

	def setParameters(self, k = 2, **parameters):

		self.k = k

	def train(self, attributeMatrix, labelVector):

		self.attributeMatrix = attributeMatrix
		self.labelVector = labelVector
		self.isTrained = True

	def predict(self, attributeVector):

		deltaMatrix = np.tile(attributeVector, (len(self.attributeMatrix),1)) - self.attributeMatrix
		distancesArray = ((deltaMatrix ** 2).sum(axis = 1)) ** 0.5
		sortedDistances = distancesArray.argsort()
		labelDict = {}
		for i in range(self.k):
			tempLabel = self.labelVector[sortedDistances[i]]
			labelDict[tempLabel] = labelDict.get(tempLabel, 0) + 1

		sortedClassCount = sorted(labelDict.items(), key = operator.itemgetter(1), reverse=True)

		return sortedClassCount[0][0]


# =================================================================================================================================

class SvmModel:

	def __init__(self):

		self.isTrained = False
		self.model = SVC()

	def train(self, attributeMatrix, labelVector):

		self.model = self.model.fit(attributeMatrix, labelVector)
		self.isTrained = True

	def predict(self, attributeVector):

		return self.model.predict([attributeVector]).tolist()[0]

	def predict_batch(self, attributeMatrix):

		return self.model.predict(attributeMatrix).tolist()

# =================================================================================================================================
# import structure
# import pybrain
# from pybrain.tools.shortcuts import buildNetwork
# from pybrain.datasets import SupervisedDataSet
# from pybrain.datasets import ClassificationDataSet
# from pybrain.utilities import percentError
# from pybrain.structure import TanhLayer
# from pybrain.structure import FeedForwardNetwork
# from pybrain.structure import LinearLayer, SigmoidLayer
# from pybrain.structure import FullConnection
# from pybrain.structure import RecurrentNetwork
# from pybrain.structure.modules import SoftmaxLayer
# from pybrain.supervised.trainers import BackpropTrainer

# class NeuralNetModel:

# 	def __init__(self):

# 		self.isTrained = False
# 		self.dataset = None
# 		self.net = None
# 		self.inputNumber = None
# 		self.outputNumber = None

# 	def toDummy(self, x):
# 		'''
# 		Function toDummy()
# 		Transfer label to dummy variables
# 		Example: toDummy(3) -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# 		'''
# 		dummy = [0, 0]
# 		dummy[x - 1] = 1
# 		return dummy

# 	def fromDummy(self, dummy):
# 		'''
# 		Function fromDummy()
# 		Transfer dummy variables to result value
# 		Example: fromDummy([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]) -> 3
# 		'''
# 		return dummy.tolist().index(max(dummy.tolist())) + 1

# 	def netBuild(self, inputNumber = 10, outputNumber = 2):
# 		'''
# 		Function netBuild()
# 		Build a singal hidden layer full connection ANN
# 		256 linear + 300 sigmoid + 10 linear
# 		'''
# 		#initialize feed forward network
# 		fnn = FeedForwardNetwork()

# 		#Create layers, including one input layer, one output layer, and some hidden layer(s) (here we only have one hidden layer)
# 		inLayer = LinearLayer(inputNumber, name = 'inLayer')
# 		hiddenLayer = SigmoidLayer(10, name = 'hiddenLayer')
# 		outLayer = LinearLayer(outputNumber, name = 'outLayer')

# 		#Add layers into the network
# 		fnn.addInputModule(inLayer)
# 		fnn.addModule(hiddenLayer)
# 		fnn.addOutputModule(outLayer)

# 		#Create connections between layers (Here we only use full connections)
# 		in_to_hidden = FullConnection(inLayer, hiddenLayer)
# 		hidden_to_out = FullConnection(hiddenLayer, outLayer)

# 		#Add connections into the network
# 		fnn.addConnection(in_to_hidden)
# 		fnn.addConnection(hidden_to_out)

# 		#Sort & prepare the network after adding so many things
# 		fnn.sortModules()

# 		return fnn

# 	'''-----------------------------------------------------------------------------'''
# 	def netBuild_DoubleHidden(self, inputNumber = 10, outputNumber = 2):
# 		'''
# 		Function netBuild_DoubleHidden()
# 		Build a double hidden layer full connection ANN
# 		256 linear + 300 sigmoid + 300 sigmoid + 10 linear
# 		'''
# 		#initialize feed forward network
# 		fnn = FeedForwardNetwork()
# 		#Create layers, including one input layer, one output layer, and some hidden layer(s)
# 		inLayer = LinearLayer(inputNumber, name = 'inLayer')
# 		hiddenLayer_1 = SigmoidLayer(10, name = 'hiddenLayer')
# 		hiddenLayer_2 = SigmoidLayer(5, name = 'hiddenLayer')
# 		outLayer = LinearLayer(outputNumber, name = 'outLayer')
# 		#Add layers into the network
# 		fnn.addInputModule(inLayer)
# 		fnn.addModule(hiddenLayer_1)
# 		fnn.addModule(hiddenLayer_2)
# 		fnn.addOutputModule(outLayer)
# 		#Create connections between layers
# 		in_to_hidden_1 = FullConnection(inLayer, hiddenLayer_1)
# 		hidden_1_to_2 = FullConnection(hiddenLayer_1, hiddenLayer_2)
# 		hidden_2_to_out = FullConnection(hiddenLayer_2, outLayer)
# 		#Add connections into the network
# 		fnn.addConnection(in_to_hidden_1)
# 		fnn.addConnection(hidden_1_to_2)
# 		fnn.addConnection(hidden_2_to_out)
# 		#Sort & prepare the network
# 		fnn.sortModules()

# 		return fnn

# 	'''-----------------------------------------------------------------------------'''
# 	def netBuild_TanhHidden(self, inputNumber = 10, outputNumber = 2):
# 		'''
# 		Function netBuild()
# 		Build a singal hidden layer full connection ANN
# 		256 linear + 300 Tanh + 10 linear
# 		'''
# 		fnn = FeedForwardNetwork()

# 		inLayer = LinearLayer(inputNumber, name = 'inLayer')
# 		hiddenLayer = TanhLayer(10, name = 'hiddenLayer')
# 		outLayer = LinearLayer(outputNumber, name = 'outLayer')

# 		fnn.addInputModule(inLayer)
# 		fnn.addModule(hiddenLayer)
# 		fnn.addOutputModule(outLayer)

# 		in_to_hidden = FullConnection(inLayer, hiddenLayer)
# 		hidden_to_out = FullConnection(hiddenLayer, outLayer)

# 		fnn.addConnection(in_to_hidden)
# 		fnn.addConnection(hidden_to_out)

# 		fnn.sortModules()

# 		return fnn

# 	'''============================================================================='''
# 	def trainNet(self, network, dataset, learningrate = 0.01, maxEpochs = 30):
# 		'''
# 		Function trainNet()
# 		Train the network
# 		'''
# 		trainer = BackpropTrainer(network, dataset, verbose = True, learningrate = learningrate)
# 		print('Training Start')
# 		trainer.trainUntilConvergence(maxEpochs = maxEpochs)
# 		return network

# 	def setParameters(self, inputNumber = 10, outputNumber = 2, learningrate = 0.01, maxEpochs = 100):

# 		self.inputNumber = inputNumber
# 		self.outputNumber = outputNumber
# 		self.learningrate = learningrate
# 		self.maxEpochs = maxEpochs
# 		self.dataset = SupervisedDataSet(inputNumber, outputNumber)
# 		self.net = self.netBuild_DoubleHidden(inputNumber = inputNumber, outputNumber = outputNumber)

# 	def train(self, attributeMatrix, labelVector):

# 		for i in range(len(attributeMatrix)):
# 			self.dataset.addSample(attributeMatrix[i], self.toDummy(labelVector[i]))

# 		self.net = self.trainNet(network = self.net, dataset = self.dataset, learningrate = self.learningrate, maxEpochs = self.maxEpochs)

# 	def predict(self, attributeVector):

# 		return self.fromDummy(self.net.activate(attributeVector))