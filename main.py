
import wave
import numpy as np
import pylab as pl
# from scipy.fftpack import fft
import time
import pygame
import os

import matplotlib.pyplot as plt
from tkinter import *
import tkinter as tk
import math, os
import easygui
import statistics
import copy
import threading
import glob
import easygui

from AudioInfo import *
from mlModel import *
from mfcc import *
import AudioInfo
import mlModel
import mfcc

from PIL import Image, ImageTk


class AudioClassifier(Frame):

	def __init__(self, master):
		Frame.__init__(self, master)
		self.master = master
		self.audioPlayer = WavPlayer()
		self.PCAentity = mlModel.PCAcontroller(PCnum = 5)
		self.selectedAudioPath = '.\\Mizuki_Nana.wav'
		self.trainSet_musicPathList = glob.glob('./trainSet/music/*.wav')
		self.trainSet_speechPathList = glob.glob('./trainSet/speech/*.wav')
		self.testSetPathList = glob.glob('./testSet/music/*.wav') + glob.glob('./testSet/speech/*.wav')
		self.modelName = None
		self.trainThread = None
		self.predictThread = None
		self.model = None
		self.isTraining = False
		self.isPredicting = False
		self.trainAttributeMatrix = None
		self.trainLabelVector = None
		self.testAttributeMatrix = None
		self.testLabelVector = None
		self.pcaFlag = True
		self.k = 2
		# =================================================================================================================
		# Create Main frame.
		self.mainFrame = Frame(master)
		self.mainFrame.pack()

		# And we have five lines here on the main frame
		self.firstLineFrame = Frame(self.mainFrame)
		self.firstLineFrame.grid(row = 0, column = 0, sticky = N)

		self.secondLineFrame = Frame(self.mainFrame)
		self.secondLineFrame.grid(row = 1, column = 0, sticky = N)

		self.thirdLineFrame = Frame(self.mainFrame)
		self.thirdLineFrame.grid(row = 2, column = 0, sticky = N)

		self.fourthLineFrame = Frame(self.mainFrame)
		self.fourthLineFrame.grid(row = 3, column = 0, sticky = N)

		# =================================================================================================================

		self.picFrame = Frame(self.firstLineFrame)
		self.picFrame.grid(row = 0, column = 0, sticky = W)

		# self.mainImageLabel = Label(master = self.picFrame, image = self.readImgFromFile(imagePath = './images/musicIcon.jpg'), height = 100, width = 100)
		self.mainImageLabel = Label(master = self.picFrame, text = '♪♫', height = 3, width = 6, font = ("Arial, 20"))
		self.mainImageLabel.pack(side = TOP)
		
		# -----------------------------------------------------------------------------------------------------------------
		self.playerFrame = Frame(self.firstLineFrame)
		self.playerFrame.grid(row = 0, column = 1, sticky = W)
		
		self.fileInfoLabel = Label(master = self.playerFrame, text = self.selectedAudioPath)
		self.fileInfoLabel.grid(row = 0, column = 0, sticky = W)

		self.playerControlFrame = Frame(master = self.playerFrame)
		self.playerControlFrame.grid(row = 1, column = 0, sticky = W)

		self.playButton = Button(master = self.playerControlFrame, text = '▶ Play', fg = 'black', padx = 10, width = 8, command = lambda: self.commandPlay())
		self.playButton.grid(row = 0, column = 0, sticky = W)

		self.stopButton = Button(master = self.playerControlFrame, text = '■ Stop', fg = 'black', padx = 10, width = 8, command = lambda: self.commandStop())
		self.stopButton.grid(row = 0, column = 1, sticky = W)

		self.viewWaveButton = Button(master = self.playerControlFrame, text = 'View Wave', fg = 'black', padx = 10, width = 8, command = lambda: self.commandViewWave())
		self.viewWaveButton.grid(row = 0, column = 2, sticky = W)

		self.openDefaultPlayerButton = Button(master = self.playerControlFrame, text = 'open in sys default player', fg = 'black', padx = 10, width = 20, command = lambda: self.commandOpenByDefaultPlayer())
		self.openDefaultPlayerButton.grid(row = 0, column = 3, sticky = W)

		# =================================================================================================================
		self.dividingLineLabel = Label(master = self.secondLineFrame, text = '---------------------------  ❄  ---------------------------')
		self.dividingLineLabel.pack()

		self.infoLabel = Label(master = self.secondLineFrame, text = '')
		self.infoLabel.pack()
		# =================================================================================================================
		# self.chooseTrainSetButton = Button(master = self.thirdLineFrame, text = 'open train set', fg = 'black', padx = 10, width = 10, command = lambda: self.commandChooseTrainSet())
		# self.chooseTrainSetButton.grid(row = 0, column = 0)

		# self.chooseTestSetButton = Button(master = self.thirdLineFrame, text = 'open test set', fg = 'black', padx = 10, width = 10, command = lambda: self.commandChooseTestSet())
		# self.chooseTestSetButton.grid(row = 0, column = 1)

		self.trainButton = Button(master = self.thirdLineFrame, text = 'Train', fg = 'black', padx = 10, width = 8, command = lambda: self.commandTrain())
		self.trainButton.grid(row = 0, column = 0)

		self.predictButton = Button(master = self.thirdLineFrame, text = 'Predict', fg = 'black', padx = 10, width = 8, command = lambda: self.commandPredict())
		self.predictButton.grid(row = 0, column = 1)

		self.modelVar = IntVar()
		self.modelVar.set(0)
		Radiobutton(self.thirdLineFrame, variable = self.modelVar, text = 'KNN Model',value = 0).grid(row = 0, column = 2)
		Radiobutton(self.thirdLineFrame, variable = self.modelVar, text = 'SVM Model',value = 1).grid(row = 0, column = 3)
		# =================================================================================================================
		self.trainSetFrame = Frame(master = self.fourthLineFrame)
		self.trainSetFrame.grid(row = 0, column = 0)

		self.trainTitleLable = Label(master = self.trainSetFrame, text = 'Train Set:')
		self.trainTitleLable.grid(row = 0, column = 0)

		self.trainListFrame = Frame(master = self.trainSetFrame)
		self.trainListFrame.grid(row = 1, column = 0)

		self.trainListScrollbar = Scrollbar(self.trainListFrame)
		self.trainListScrollbar.pack(side = RIGHT, fill = Y)

		self.trainListBox = Listbox(self.trainListFrame, yscrollcommand = self.trainListScrollbar.set, selectmode = BROWSE, height = 7)
		i = 0
		for tempPath in self.trainSet_musicPathList:
			self.trainListBox.insert(i, tempPath)
			i += 1
		for tempPath in self.trainSet_speechPathList:
			self.trainListBox.insert(i, tempPath)
			i += 1
		self.trainListBox.pack(side = LEFT, fill = BOTH)
		self.trainListBox.activate(1)
		self.trainListBox.bind('<<ListboxSelect>>', self.commandSelectAudioFromListBox_trainSet)
		self.trainListScrollbar.config(command = self.trainListBox.yview)
		# -----------------------------------------------------------------------------------------------------------------
		self.testSetFrame = Frame(master = self.fourthLineFrame)
		self.testSetFrame.grid(row = 0, column = 1)

		self.testTitleLable = Label(master = self.testSetFrame, text = 'Test Set:')
		self.testTitleLable.grid(row = 0, column = 0)

		self.testListFrame = Frame(master = self.testSetFrame)
		self.testListFrame.grid(row = 1, column = 0)

		self.testListScrollbar = Scrollbar(self.testListFrame)
		self.testListScrollbar.pack(side = RIGHT, fill = Y)

		self.testListBox = Listbox(self.testListFrame, yscrollcommand = self.testListScrollbar.set, selectmode = BROWSE, height = 7)
		i = 0
		for tempPath in self.testSetPathList:
			self.testListBox.insert(i, tempPath)
			i += 1
		self.testListBox.pack(side = LEFT, fill = BOTH)
		self.testListBox.activate(1)
		self.testListBox.bind('<<ListboxSelect>>', self.commandSelectAudioFromListBox_testSet)
		self.testListScrollbar.config(command = self.testListBox.yview)
		# -----------------------------------------------------------------------------------------------------------------
		self.musicSetFrame = Frame(master = self.fourthLineFrame)
		self.musicSetFrame.grid(row = 0, column = 2)

		self.musicTitleLable = Label(master = self.musicSetFrame, text = 'Music:')
		self.musicTitleLable.grid(row = 0, column = 0)

		self.musicListFrame = Frame(master = self.musicSetFrame)
		self.musicListFrame.grid(row = 1, column = 0)

		self.musicListScrollbar = Scrollbar(self.musicListFrame)
		self.musicListScrollbar.pack(side = RIGHT, fill = Y)

		self.musicListBox = Listbox(self.musicListFrame, yscrollcommand = self.musicListScrollbar.set, selectmode = BROWSE, height = 7)
		self.musicListBox.pack(side = LEFT, fill = BOTH)
		self.musicListBox.activate(1)
		self.musicListBox.bind('<<ListboxSelect>>', self.commandSelectAudioFromListBox_musicSet)
		self.musicListScrollbar.config(command = self.musicListBox.yview)
		# -----------------------------------------------------------------------------------------------------------------
		self.speechSetFrame = Frame(master = self.fourthLineFrame)
		self.speechSetFrame.grid(row = 0, column = 3)

		self.speechTitleLable = Label(master = self.speechSetFrame, text = 'Speech:')
		self.speechTitleLable.grid(row = 0, column = 0)

		self.speechListFrame = Frame(master = self.speechSetFrame)
		self.speechListFrame.grid(row = 1, column = 0)

		self.speechListScrollbar = Scrollbar(self.speechListFrame)
		self.speechListScrollbar.pack(side = RIGHT, fill = Y)

		self.speechListBox = Listbox(self.speechListFrame, yscrollcommand = self.speechListScrollbar.set, selectmode = BROWSE, height = 7)
		self.speechListBox.pack(side = LEFT, fill = BOTH)
		self.speechListBox.activate(1)
		self.speechListBox.bind('<<ListboxSelect>>', self.commandSelectAudioFromListBox_speechSet)
		self.speechListScrollbar.config(command = self.speechListBox.yview)
		# =================================================================================================================
		# root.mainloop()
		# =================================================================================================================

	def readImgFromFile(self, imagePath):

		im = Image.open(imagePath)
		imSize = im.size
		x = imSize[0]/2
		y = imSize[1]/2
		imX = copy.deepcopy(im)
		imResize = imX.resize((int(x), int(y)), Image.ANTIALIAS)
		img = ImageTk.PhotoImage(imResize)
		return img


		# im = Image.open(imagePath)
		# imResized = im.resize((int(im.size[0]/2), int(im.size[1]/2)), Image.ANTIALIAS)
		# return ImageTk.PhotoImage(im)

	def setSelectedAudioInfo(self, text):
		self.fileInfoLabel.configure(text = text)

	def setMainInfoLabel(self, text):
		self.infoLabel.configure(text = text)

	def prepareAttributeAndLabel_batch(self, filePathList = None):

		filePathList = self.trainSet_musicPathList + self.trainSet_speechPathList if filePathList == None else filePathList

		attributeMatrix = []
		labelVector = []

		for tempFilePath in filePathList:
			print('Info: handling file', tempFilePath)
			tempAudioData = AudioInfo.AudioData(filePath = tempFilePath)
			tempAttributeVector = tempAudioData.getAttributeVector()
			tempLabel = tempAudioData.getLabel()
			attributeMatrix.append(tempAttributeVector)
			labelVector.append(tempLabel)
			# print('Info: audio attribute =', tempAttributeVector, '; label =', tempLabel)

		return attributeMatrix, labelVector


	# =======================================================================================================================================================

	def commandPlay(self):
		if pygame.mixer.music.get_busy() == False:
			self.audioPlayer.play(filePath = self.selectedAudioPath, postStopFunction = lambda: self.playButton.configure(text = '▶ Play'))
			self.playButton.configure(text = '❄ Pause')
		elif self.audioPlayer.isPause == False:
			self.audioPlayer.pause()
			self.playButton.configure(text = '▶ Play')
		else:
			self.audioPlayer.unpause()
			self.playButton.configure(text = '❄ Pause')

	def commandStop(self):
		self.audioPlayer.stop()
		self.playButton.configure(text = '▶ Play')

	def commandViewWave(self):
		AudioData(filePath = self.selectedAudioPath).doFftAndDrawGraph()

	def commandOpenByDefaultPlayer(self):
		os.startfile(self.selectedAudioPath)

	def selectAudioFromListBox(self, listBox):
		if len(listBox.curselection()) == 0:
			return
		currentSelectedIndex = listBox.curselection()[0]
		filePath = listBox.get(first = currentSelectedIndex, last = currentSelectedIndex)[0]
		self.selectedAudioPath = filePath
		self.setSelectedAudioInfo(text = filePath)
		self.commandStop()

	def commandSelectAudioFromListBox_trainSet(self, event):
		self.selectAudioFromListBox(listBox = self.trainListBox)

	def commandSelectAudioFromListBox_testSet(self, event):
		self.selectAudioFromListBox(listBox = self.testListBox)

	def commandSelectAudioFromListBox_musicSet(self, event):
		self.selectAudioFromListBox(listBox = self.musicListBox)

	def commandSelectAudioFromListBox_speechSet(self, event):
		self.selectAudioFromListBox(listBox = self.speechListBox)

	def commandChooseTrainSet(self):
		pass

	def commandChooseTestSet(self):
		pass

	def commandTrain(self):

		print('Info: command train')
		
		self.isTraining = True
		modelIndex = self.modelVar.get()

		if modelIndex == 0:
			self.modelName = 'KNN'
			self.model = mlModel.KnnModel(k = self.k)
		elif modelIndex == 1:
			self.modelName = 'SVM'
			self.model = mlModel.SvmModel()
		else:
			self.modelName = 'UNDEFINED'

		print('Info: use model', self.modelName)
		self.setMainInfoLabel(text = 'Training Start. Use model ' + self.modelName + '.\nPlease wait...')

		self.trainThread = threading.Thread(target = self.train, args = ())
		self.trainThread.setDaemon(True)
		self.trainThread.start()


	def commandPredict(self):
		print('Info: command predict')

		if self.model == None:
			self.commandTrain()
			return
		
		self.isPredicting = True

		self.setMainInfoLabel(text = 'Prediction Start. Use model ' + self.modelName + '.\nPlease wait...')

		self.predictThread = threading.Thread(target = self.predict, args = ())
		self.predictThread.setDaemon(True)
		self.predictThread.start()
		


	# =======================================================================================================================================================

	def train(self):

		self.trainAttributeMatrix, self.trainLabelVector = self.prepareAttributeAndLabel_batch(filePathList = self.trainSet_musicPathList + self.trainSet_speechPathList)
		# print('Debug: self.trainAttributeMatrix[0] =', self.trainAttributeMatrix[0])
		reconstructedMatrix = self.PCAentity.PCA_train(matrix = self.trainAttributeMatrix, pcaFlag = self.pcaFlag)
		self.model.train(attributeMatrix = reconstructedMatrix, labelVector = self.trainLabelVector)

		self.setMainInfoLabel(text = 'Training Finish. Successfully build ' + self.modelName + ' model on training set.\nPlease press [predict] to move on.')
		self.isTraining = False

	# =======================================================================================================================================================

	def predict(self):

		self.testAttributeMatrix, self.testLabelVector = self.prepareAttributeAndLabel_batch(filePathList = self.testSetPathList)

		confusionMatrix = [[0, 0, 0], [0, 0, 0]]

		# help(self.musicListBox)
		# zz = self.musicListBox.get(first = 0, last = END)
		# print('zz1 =',zz)
		self.musicListBox.delete(first = 0, last = END)
		self.speechListBox.delete(first = 0, last = END)
		# zz = self.musicListBox.get(first = 0, last = END)
		# print('zz2 =', zz)
		# mipa

		for tempAttributeVector, tempLabel, tempPath, i in zip(self.testAttributeMatrix, self.testLabelVector, self.testSetPathList, list(range(len(self.testSetPathList)))):
			# print('Debug: tempAttributeVector =', tempAttributeVector)
			predictValue = self.model.predict(attributeVector = self.PCAentity.PCA_predict(vector = tempAttributeVector, pcaFlag = self.pcaFlag))
			if predictValue == 'MUSIC':
				self.musicListBox.insert(END, tempPath)
			elif predictValue == 'SPEECH':
				self.speechListBox.insert(END, tempPath)
			else:
				print('Error: unknow predicted class from model, predictValue =', predictValue)

			if predictValue == tempLabel:
				print('Info: (v) prediction of audio', tempPath, 'is correct.')
				if tempLabel == 'MUSIC':
					confusionMatrix[0][0] += 1
				elif tempLabel == 'SPEECH':
					confusionMatrix[1][1] += 1
				else:
					pass
			elif tempLabel == 'UNKNOW':
				print('Info: (?) unlabeled test data.')
				if predictValue == 'MUSIC':
					confusionMatrix[0][2] += 1
				elif predictValue == 'SPEECH':
					confusionMatrix[1][2] += 1
				else:
					pass
			else:
				print('Info: (x) prediction of audio', tempPath, 'is incorrect.')
				if tempLabel == 'MUSIC':
					confusionMatrix[1][0] += 1
				elif tempLabel == 'SPEECH':
					confusionMatrix[0][1] += 1
				else:
					pass

		additionalColumn = ['', '', ''] if confusionMatrix[0][2] == 0 and confusionMatrix[1][2] == 0 else ['\tUnlabeled', '\t' + str(confusionMatrix[0][2]), '\t' + str(confusionMatrix[1][2])]

		print('================================================================')
		print('Confusion Matrix')
		print('P\\A\tMusic\tSpeech' + additionalColumn[0])
		print('Music\t' + str(confusionMatrix[0][0]) + '\t' + str(confusionMatrix[0][1]) + additionalColumn[1])
		print('Speech\t' + str(confusionMatrix[1][0]) + '\t' + str(confusionMatrix[1][1]) + additionalColumn[2])
		# self.musicListBox.insert(0, './test.wav')
		# self.speechListBox.insert(0, './test2.wav')
		self.setMainInfoLabel(text = 'Prediction Finish.\nClick on file name to load audio into player.')
		self.isPredicting = False

	


# =============================================================================================================================================================
# =============================================================================================================================================================
# =============================================================================================================================================================

if __name__ == '__main__':
	root = Tk()
	root.title('Audio Analysis Tool')
	mainClassifier = AudioClassifier(root)
	root.mainloop()

