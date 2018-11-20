# ===================================================================================================================================
# from tkinter import *
# import wave
# import matplotlib.pyplot as plt
# import numpy as np
 
# def read_wave_data(file_path):
# 	#open a wave file, and return a Wave_read object
# 	f = wave.open(file_path,"rb")
# 	#read the wave's format infomation,and return a tuple
# 	params = f.getparams()
# 	#get the info
# 	nchannels, sampwidth, framerate, nframes = params[:4]
# 	#Reads and returns nframes of audio, as a string of bytes. 
# 	str_data = f.readframes(nframes)
# 	#close the stream
# 	f.close()
# 	#turn the wave's data to array
# 	wave_data = np.fromstring(str_data, dtype = np.short)
# 	#for the data is stereo,and format is LRLRLR...
# 	#shape the array to n*2(-1 means fit the y coordinate)
# 	wave_data.shape = -1, 2
# 	#transpose the data
# 	wave_data = wave_data.T
# 	#calculate the time bar
# 	time = np.arange(0, nframes) * (1.0/framerate)
# 	return wave_data, time
 
# def main():
# 	wave_data, time = read_wave_data('./test.wav')	
# 	#draw the wave
# 	plt.subplot(211)
# 	plt.plot(time, wave_data[0])
# 	plt.subplot(212)
# 	plt.plot(time, wave_data[1], c = "g")
# 	plt.show()
 
# if __name__ == "__main__":
# 	main()
# ===================================================================================================================================

# import time
# import pygame
# from multiprocessing import Process
# import multiprocessing
# import threading

# def play():
# 	file = './test.wav'  # mp3文件路径
# 	pygame.mixer.init()
# 	print("Playing",file)
# 	track = pygame.mixer.music.load(file)
# 	pygame.mixer.music.play()
# 	while True:
# 		time.sleep(1)
# 		if pygame.mixer.music.get_busy() == False:
# 			break



# # time.sleep(32)                   #播放时间
# # pygame.mixer.music.stop()

# # track = pygame.mixer.music.load('./test2.wav')
# # pygame.mixer.music.play()
# # time.sleep(32)
# def foo():
# 	print('hh')


# if __name__ == '__main__':
# 	multiprocessing.freeze_support()
# 	pygame.mixer.init()
# 	p1 = Process(target = play)
# 	p1.start()

# 	time.sleep(10)
# 	print('emmmm')
# 	pygame.mixer.music.pause()
# 	pygame.mixer.music.stop()

# =====================================================================================================================================================

# import numpy as np
# from scipy.fftpack import fft,ifft
# import matplotlib.pyplot as plt


# #采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
# x=np.linspace(0,1,1400)      

# #设置需要采样的信号，频率分量有180，390和600
# y=7*np.sin(2*np.pi*180*x) + 2.8*np.sin(2*np.pi*390*x)+5.1*np.sin(2*np.pi*600*x)

# yy=fft(y)                     #快速傅里叶变换
# yreal = yy.real               # 获取实数部分
# yimag = yy.imag               # 获取虚数部分

# yf=abs(fft(y))                # 取绝对值
# yf1=abs(fft(y))/len(x)           #归一化处理
# yf2 = yf1[range(int(len(x)/2))]  #由于对称性，只取一半区间

# xf = np.arange(len(y))        # 频率
# xf1 = xf
# xf2 = xf[range(int(len(x)/2))]  #取一半区间


# plt.subplot(221)
# plt.plot(x[0:50],y[0:50])   
# plt.title('Original wave')


# plt.subplot(222)
# plt.plot(xf,yf,'r')
# plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表

# plt.subplot(223)
# plt.plot(xf1,yf1,'g')
# plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')

# plt.subplot(224)
# plt.plot(xf2,yf2,'b')
# plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')


# plt.show()
# --------------------------------------------------------------------------------------------------------------------------------------

# from pylab import *
# import numpy as np
# import matplotlib.pyplot as plt

# nSampleNum = 5120
# ncount = 2048.0
# df = nSampleNum / ncount
# sampleTime = ncount / nSampleNum
# freqLine = 800

# x = np.linspace(0,sampleTime,ncount)#时域波形x轴坐标
# sinx = np.sin(2*pi*250*x)
# sinx2 = 0.5*np.sin(2*pi*500*x)
# sinx3 = 0.3*np.sin(2*pi*1000*x)    #以上是三个标准正弦波形

# sinx += sinx2
# sinx += sinx3  #叠加一个时域波形

# fft = np.fft.fft(sinx)[0:freqLine]  #调用fft变换算法计算频域波形
# fftx = np.linspace(0,df*freqLine,freqLine)  #频域波形x轴坐标311)

# plt.subplot(211)
# plt.plot(x,sinx)
# plt.xlabel('time(s)')
# plt.ylabel('amplitude')
# plt.title('time domain graph')

# plt.subplot(212)
# plt.plot(fftx,abs(fft))
# plt.xlabel('freqency(Hz)')
# plt.ylabel('amplitude')
# plt.title('frequency domain graph')

# plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------



# import numpy as np
# import pylab as pl

# sampling_rate = 8000
# fft_size = 512
# t = np.arange(0, 1.0, 1.0/sampling_rate)
# x = np.sin(2*np.pi*156.25*t)  + 2*np.sin(2*np.pi*234.375*t)
# xs = x[:fft_size]
# xf = np.fft.rfft(xs)/fft_size
# freqs = np.linspace(0, sampling_rate/2, fft_size/2+1)
# xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))

# pl.figure(figsize=(8,4))
# pl.subplot(211)
# pl.plot(t[:fft_size], xs)
# pl.xlabel('time(s)')
# pl.title('156.25Hz & 234.375Hz')
# pl.subplot(212)
# pl.plot(freqs, xfp)
# pl.xlabel('frequency(Hz)')
# pl.subplots_adjust(hspace=0.4)
# pl.show()

# -------------------------------------------------------------------------------------------------------------------------------------------------------

# import wave
# import pyaudio
# import numpy
# import pylab

# #打开WAV文档，文件路径根据需要做修改
# wf = wave.open('./test.wav', "rb")
# #创建PyAudio对象
# p = pyaudio.PyAudio()
# stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
# nframes = wf.getnframes()
# framerate = wf.getframerate()
# #读取完整的帧数据到str_data中，这是一个string类型的数据
# str_data = wf.readframes(nframes)
# wf.close()
# #将波形数据转换为数组
# # A new 1-D array initialized from raw binary or text data in a string.
# wave_data = numpy.fromstring(str_data, dtype=numpy.short)
# #将wave_data数组改为2列，行数自动匹配。在修改shape的属性时，需使得数组的总长度不变。
# wave_data.shape = -1,2 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# #将数组转置
# wave_data = wave_data.T
# #time 也是一个数组，与wave_data[0]或wave_data[1]配对形成系列点坐标
# #time = numpy.arange(0,nframes)*(1.0/framerate)
# #绘制波形图
# #pylab.plot(time, wave_data[0])
# #pylab.subplot(212)
# #pylab.plot(time, wave_data[1], c="g")
# #pylab.xlabel("time (seconds)")
# #pylab.show()
# #
# # 采样点数，修改采样点数和起始位置进行不同位置和长度的音频波形分析
# N=44100
# start=0 #开始采样位置
# df = framerate/(N-1) # 分辨率
# freq = [df*n for n in range(0,N)] #N个元素
# wave_data2=wave_data[0][start:start+N]
# c=numpy.fft.fft(wave_data2)*2/N
# #常规显示采样频率一半的频谱
# d=int(len(c)/2)
# #仅显示频率在4000以下的频谱
# while freq[d]>4000:
# 	d -= 10
# pylab.plot(freq[:d-1],abs(c[:d-1]),'r')
# pylab.show()

# =================================================================================================================================================================

# import time
# import threading
# import pygame

# def f0():
# 	track = pygame.mixer.music.load('./test.wav')
# 	pygame.mixer.music.play()
 
# def f1():
# 	pygame.mixer.music.pause()
# 	print('Haha!') if pygame.mixer.music.get_busy() == False else print('emmmmmm')

# pygame.mixer.init()


# # t= threading.Thread(target=f1,args=(111,112))#创建线程
# # t.setDaemon(True)#设置为后台线程，这里默认是False，设置为True之后则主线程不用等待子线程
# # t.start()#开启线程
 
# t0 = threading.Thread(target=f0)
# t0.start()

# time.sleep(5)

# t1 = threading.Thread(target=f1)
# t1.start()
# =========================================================================================================================================================================

# from PIL import Image,ImageTk
# import tkinter as tk

# # 简单插入显示
# def show_jpg():
#     root = tk.Tk()
#     im=Image.open("musicIcon.jpg")
#     img=ImageTk.PhotoImage(im)
#     imLabel=tk.Label(root,image=img).pack()
#     root.mainloop()

# if __name__ == '__main__':
#     show_jpg()

# ============================================================

from PIL import Image, ImageTk
import tkinter as tk


def f(root):

	# root = tk.Tk()
	im = Image.open("musicIcon.jpg")
	# img = ImageTk.PhotoImage(file = 'music.gif')
	img = ImageTk.PhotoImage(im)
	tk.Label(root, image = img).pack()
	# root.mainloop()
	return root
	
	# -----------------------------------------------------------------------------------------------------------------
		
if __name__ == '__main__':
	root = tk.Tk()
	print(type(root))
	print(root)
	rr = f(root)
	print(type(rr))
	print(rr)
	rr.mainloop()

	# root.mainloop()

