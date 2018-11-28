
import wave
import numpy as np
import pylab as pl
from scipy.fftpack import fft

import time
import pygame
import threading

from tkinter import *
import matplotlib.pyplot as plt

import scipy.io.wavfile as wav
import numpy
import mfcc


# def XOR(a, b):
# 	return ((not a) and b) or (a and (not b))

def sng_batch(ndarray_x):
	return (ndarray_x > 0).astype('int') - (ndarray_x < 0).astype('int')


class AudioData():

	def __init__(self, filePath = './test.wav'):

		self.filePath = filePath
		self.nchannels = None
		self.sampwidth = None
		self.framerate = None
		self.nframes = None
		self.comptype = None
		self.compname = None
		self.waveData = None
		self.frequencyData = None
		self.frequencyGraph_X = None
		self.frequencyGraph_Y = None
		self.timePointList = None
		self.averageEnergy = None
		self.zeroCrossingRate = None
		self.binList = []
		self.attributeList = []
		self.label = None

		self.readAudioFile(filePath = filePath)

	def findDirectFolder(self, path):

		xList = path.split('/')
		sList = []
		for xStr in xList:
			sList += xStr.split('\\')
		return sList[-2]

	def readAudioFile(self, filePath = None):

		filePath = self.filePath if filePath == None else filePath

		self.label = 'MUSIC' if self.findDirectFolder(path = filePath) == 'music' else 'SPEECH' if self.findDirectFolder(path = filePath) == 'speech' else 'UNKNOW'

		with wave.open(filePath,'rb') as audioFile:
			audioParams = audioFile.getparams()
			# print(audioParams)
			
			self.nchannels, self.sampwidth, self.framerate, self.nframes, self.comptype, self.compname = audioParams
			rawData = audioFile.readframes(self.nframes)

		waveData = np.fromstring(rawData, dtype = np.short)
		waveData.shape = -1, self.nchannels
		self.waveData = waveData.T
		self.timePointList = np.arange(0, self.nframes) * (1.0/self.framerate)

		return (self.waveData, self.timePointList)

	# ==========================================================================================================

	def calculateAverageEnergy(self, waveData = None):
		
		waveData = self.waveData if waveData == None else waveData

		self.averageEnergy = np.mean([np.sum(waveData[i] ** 2)/len(waveData[i]) for i in range(self.nchannels)])
		# print('Info: averageEnergy of', self.filePath, 'is:', str(self.averageEnergy))

		return self.averageEnergy

	def calculateZeroCrossingRate(self, waveData = None):

		waveData = self.waveData if waveData == None else waveData

		self.zeroCrossingRate = np.mean([sum(abs(sng_batch(waveData[i][1:]) - sng_batch(waveData[i][:-1])))/(2*(len(waveData[i]) - 1)) for i in range(self.nchannels)])
		# print('Info: zeroCrossingRate of', self.filePath, 'is:', str(self.zeroCrossingRate))

		return self.zeroCrossingRate

	# def doFFT_versionA(self, waveData = None):
		
	# 	waveData = self.waveData[0] if waveData == None else waveData

	# 	self.frequencyData = np.fft.rfft(waveData)/len(waveData)
	# 	X = np.linspace(0, self.framerate/2, num = len(waveData)/2 + 1)
	# 	Y = 20 * np.log10(np.clip(np.abs(self.frequencyData), 1e-20, 1e100))

	# 	return X, Y

	def doFFT(self, waveData = None):

		waveData = self.waveData[0] if waveData == None else waveData
		# startPos = 0 #开始采样位置
		# endPos = len(waveData)
		# df = self.framerate/(endPos - startPos) # 分辨率
		# freq = [df*n for n in range(0,endPos)] #N个元素
		# waveData = waveData[startPos: endPos]
		# self.frequencyData = np.fft.fft(waveData)/(endPos - startPos)
		# X = freq[:int(len(self.frequencyData)/2)]
		# Y = abs(self.frequencyData[:int(len(self.frequencyData)/2)])
		self.frequencyData = np.fft.fft(waveData)/len(waveData)
		self.frequencyGraph_X = np.linspace(0, self.framerate/2, num = len(waveData)/2)
		self.frequencyGraph_Y = abs(self.frequencyData[:int(len(self.frequencyData)/2)])
		# print('Info: successfully generate frequency graph.')

		return self.frequencyGraph_X, self.frequencyGraph_Y
		# return self.frequencyGraph_X[:int(len(self.frequencyGraph_X)/4)], self.frequencyGraph_Y[:int(len(self.frequencyGraph_X)/4)]


	# def doFFT_versionC(self, waveData = None):

	# 	waveData = self.waveData[0] if waveData == None else waveData

	# 	print('Info: FFT start')
	# 	self.frequencyGraph_Y = (abs(fft(waveData))/self.nframes)[range(int(self.nframes/2))]
	# 	# self.frequencyGraph_Y = (abs(np.fft.fft(waveData))/self.nframes)[range(int(self.nframes/2))]
	# 	self.frequencyGraph_X = np.arange(len(waveData))[range(int(self.nframes/2))]
	# 	print('Info: FFT finish')

	# 	return self.frequencyGraph_X, self.frequencyGraph_Y

	def drawFrequencyGraph(self):

		xCount = 0
		for (tempX, tempY) in zip(self.frequencyGraph_X, self.frequencyGraph_Y):
			xCount += 1
			if tempX >= 6000:
				break
		
		plt.subplot(311)
		plt.plot(range(len(self.waveData[0])), self.waveData[0], c = 'b')
		plt.subplot(312)
		plt.plot(self.frequencyGraph_X[:xCount], self.frequencyGraph_Y[:xCount], c = 'g')
		plt.subplot(313)
		plt.bar(list(range(len(self.binList))), self.binList)
		plt.show()

	def calculateFrequencyHistogram(self):

		binWidthList = [500] * 2 + [1000] * 7
		binWidthList.reverse()
		binWidthListLength = len(binWidthList)
		self.binList = []
		currentBin = 0
		totalBin = 0
		binCountFlag = True

		currentBinMax = binWidthList.pop()
		for currentX, currentY in zip(self.frequencyGraph_X, self.frequencyGraph_Y):
			totalBin += currentY
			if currentX > currentBinMax and binCountFlag:
				self.binList.append(currentBin)
				try:
					currentBinMax += binWidthList.pop()
				except:
					binCountFlag = False
				currentBin = currentY
			else:
				currentBin += currentY
		self.binList.append(currentBin)

		while len(self.binList) < binWidthListLength:
			self.binList.append(0)
		self.binList = self.binList[:binWidthListLength]

		self.binList = [tempBin/totalBin for tempBin in self.binList]

		# print(self.binList)

		return self.binList

	def calculateBandwidth(self):
		pass

	def calculateMFCC(self, windowNum = 1):
		
		(rate,sig) = wav.read(self.filePath)
		# print(len(sig), rate)
		mfcc_feat = mfcc.calcMFCC_delta_delta(signal = sig, samplerate = rate, win_length = (len(sig)/rate)/windowNum)
		# print(mfcc_feat.shape)

		mfccVector = []
		for i in range(len(mfcc_feat)):
			mfccVector += [float(tempValue) for tempValue in list(mfcc_feat[i])][:13]

		# print(mfccVector)
		return mfccVector

	def doFftAndDrawGraph(self):

		self.doFFT()
		self.calculateFrequencyHistogram()
		self.drawFrequencyGraph()

		
	
	# ==========================================================================================================
	
	def getLabel(self):
		
		return self.label

	def getAttributeVector(self):

		def cheat():
			return 0 if self.label == 'MUSIC' else 1
		
		self.doFFT()
		# self.attributeList = [self.calculateAverageEnergy(), self.calculateZeroCrossingRate(), self.sampwidth, self.framerate] + self.calculateFrequencyHistogram()
		# self.attributeList = [cheat(), cheat(), 0]
		self.attributeList = self.calculateFrequencyHistogram()
		# self.attributeList += self.calculateMFCC()

		# print(self.attributeList)

		return self.attributeList


	def testFunc(self):
		# (waveData, timePointList) = self.readAudioFile('./test.wav')	
		# #draw the wave
		# plt.subplot(211)
		# plt.plot(timePointList, waveData[0])
		# plt.subplot(212)
		# plt.plot(timePointList, waveData[1], c = "g")
		# plt.show()
		# ---------------------------------------------------------------------------
		self.calculateAverageEnergy()
		self.calculateZeroCrossingRate()
		self.doFFT()
		self.drawFrequencyGraph()
		# plt.subplot(211)
		# plt.plot(range(len(self.waveData[0])), self.waveData[0], c = 'b')
		# plt.subplot(212)
		# plt.plot(X[:], Y[:], c = 'g')
		# plt.show()
		# ---------------------------------------------------------------------------
		pass


# AudioData().testFunc()


class WavPlayer:

	def __init__(self):
		self.status = 'STOP'
		self.playThread = None
		self.isPause = False
		pygame.mixer.init()

	def playAudio(self, filePath = './test.wav', startPosition = None, postStopFunction = None):

		track = pygame.mixer.music.load(filePath)
		pygame.mixer.music.set_pos(startPosition) if startPosition != None else None

		pygame.mixer.music.play()

		if postStopFunction != None:
			while True:
				time.sleep(1)
				if pygame.mixer.music.get_busy() == False:
					postStopFunction()

	def play(self, filePath = './test.wav', startPosition = None, postStopFunction = None):

		self.playThread = threading.Thread(target = self.playAudio, args = (filePath, startPosition, postStopFunction))
		self.playThread.setDaemon(True)
		self.playThread.start()
	
	def stop(self):
		pygame.mixer.music.stop()

	def pause(self):
		pygame.mixer.music.pause()
		self.isPause = True

	def unpause(self):
		pygame.mixer.music.unpause()
		self.isPause = False

	def restart(self):
		pygame.mixer.music.rewind()