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

# from PIL import Image, ImageTk
# import tkinter as tk


# def f(root):

# 	# root = tk.Tk()
# 	im = Image.open("musicIcon.jpg")
# 	# img = ImageTk.PhotoImage(file = 'music.gif')
# 	img = ImageTk.PhotoImage(im)
# 	tk.Label(root, image = img).pack()
# 	# root.mainloop()
# 	return root
	
# 	# -----------------------------------------------------------------------------------------------------------------
		
# if __name__ == '__main__':
# 	root = tk.Tk()
# 	print(type(root))
# 	print(root)
# 	rr = f(root)
# 	print(type(rr))
# 	print(rr)
# 	rr.mainloop()

# 	# root.mainloop()

# =============================================================================================================================================================

# import numpy as np
# from pydub import AudioSegment
# import pydub
# import os
# import wave
# import json
# from matplotlib import pyplot as plt

# def MP32WAV(mp3_path,wav_path):
#     """
#     这是MP3文件转化成WAV文件的函数
#     :param mp3_path: MP3文件的地址
#     :param wav_path: WAV文件的地址
#     """
#     pydub.AudioSegment.converter = "D:\\software_installed\\ffmpeg\\bin\\ffmpeg.exe"            #说明ffmpeg的地址
#     print(mp3_path)
#     MP3_File = AudioSegment.from_mp3(file = '.\\mp3\\test3.mp3')
#     MP3_File.export(wav_path, format="wav")

# def Read_WAV(wav_path):
#     """
#     这是读取wav文件的函数，音频数据是单通道的。返回json
#     :param wav_path: WAV文件的地址
#     """
#     wav_file = wave.open(wav_path,'r')
#     numchannel = wav_file.getnchannels()          # 声道数
#     samplewidth = wav_file.getsampwidth()      # 量化位数
#     framerate = wav_file.getframerate()        # 采样频率
#     numframes = wav_file.getnframes()           # 采样点数
#     print("channel", numchannel)
#     print("sample_width", samplewidth)
#     print("framerate", framerate)
#     print("numframes", numframes)
#     Wav_Data = wav_file.readframes(numframes)
#     Wav_Data = np.fromstring(Wav_Data,dtype=np.int16)
#     Wav_Data = Wav_Data*1.0/(max(abs(Wav_Data)))        #对数据进行归一化
#     # 生成音频数据,ndarray不能进行json化，必须转化为list，生成JSON
#     dict = {"channel":numchannel,
#             "samplewidth":samplewidth,
#             "framerate":framerate,
#             "numframes":numframes,
#             "WaveData":list(Wav_Data)}
#     return json.dumps(dict)

# def DrawSpectrum(wav_data,framerate):
#     """
#     这是画音频的频谱函数
#     :param wav_data: 音频数据
#     :param framerate: 采样频率
#     """
#     Time = np.linspace(0,len(wav_data)/framerate*1.0,num=len(wav_data))
#     plt.figure(1)
#     plt.plot(Time,wav_data)
#     plt.grid(True)
#     plt.show()
#     plt.figure(2)
#     Pxx, freqs, bins, im = plt.specgram(wav_data, NFFT=1024, Fs = 16000, noverlap=900)
#     plt.show()
#     print(Pxx)
#     print(freqs)
#     print(bins)
#     print(im)

# def run_main():
#     """
#         这是主函数
#     """
#     # MP3文件和WAV文件的地址
#     path1 = './mp3'
#     path2 = './wav'
#     paths = os.listdir(path1)
#     mp3_paths = []
#     # 获取mp3文件的相对地址
#     for mp3_path in paths:
#         mp3_paths.append(path1+"/"+mp3_path)
#     print(mp3_paths)

#     # 得到MP3文件对应的WAV文件的相对地址
#     wav_paths = []
#     for mp3_path in mp3_paths:
#        wav_path = path2+"/"+mp3_path[1:].split('.')[0].split('/')[-1]+'.wav'
#        wav_paths.append(wav_path)
#     print(wav_paths)

#     # 将MP3文件转化成WAV文件
#     for(mp3_path,wav_path) in zip(mp3_paths,wav_paths):
#         # MP32WAV(mp3_path,wav_path)
#         pass


#     wav_paths = ['./test.wav']
#     for wav_path in wav_paths:
#         Read_WAV(wav_path)

#     # 开始对音频文件进行数据化
#     for wav_path in wav_paths:
#         wav_json = Read_WAV(wav_path)
#         # print(wav_json)
#         wav = json.loads(wav_json)
#         wav_data = np.array(wav['WaveData'])
#         framerate = int(wav['framerate'])
#         DrawSpectrum(wav_data,framerate)

# if __name__ == '__main__':
#     run_main()
# ===============================================================================================================================================================


#coding=utf-8
#对音频信号处理程序
#张泽旺，2015-12-12
# 本程序主要有四个函数，它们分别是：
#    audio2frame:将音频转换成帧矩阵
#    deframesignal:对每一帧做一个消除关联的变换
#    spectrum_magnitude:计算每一帧傅立叶变换以后的幅度
#    spectrum_power:计算每一帧傅立叶变换以后的功率谱
#    log_spectrum_power:计算每一帧傅立叶变换以后的对数功率谱
#    pre_emphasis:对原始信号进行预加重处理
import numpy
import math

def audio2frame(signal,frame_length,frame_step,winfunc=lambda x:numpy.ones((x,))):
    '''将音频信号转化为帧。
	参数含义：
	signal:原始音频型号
	frame_length:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
	frame_step:相邻帧的间隔（同上定义）
	winfunc:lambda函数，用于生成一个向量
    '''
    signal_length=len(signal) #信号总长度
    frame_length=int(round(frame_length)) #以帧帧时间长度
    frame_step=int(round(frame_step)) #相邻帧之间的步长
    if signal_length<=frame_length: #若信号长度小于一个帧的长度，则帧数定义为1
        frames_num=1
    else: #否则，计算帧的总长度
        frames_num=1+int(math.ceil((1.0*signal_length-frame_length)/frame_step))
    pad_length=int((frames_num-1)*frame_step+frame_length) #所有帧加起来总的铺平后的长度
    zeros=numpy.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal=numpy.concatenate((signal,zeros)) #填补后的信号记为pad_signal
    indices=numpy.tile(numpy.arange(0,frame_length),(frames_num,1))+numpy.tile(numpy.arange(0,frames_num*frame_step,frame_step),(frame_length,1)).T  #相当于对所有帧的时间点进行抽取，得到frames_num*frame_length长度的矩阵
    indices=numpy.array(indices,dtype=numpy.int32) #将indices转化为矩阵
    frames=pad_signal[indices] #得到帧信号
    win=numpy.tile(winfunc(frame_length),(frames_num,1))  #window窗函数，这里默认取1
    return frames*win   #返回帧信号矩阵

def deframesignal(frames,signal_length,frame_length,frame_step,winfunc=lambda x:numpy.ones((x,))):
    '''定义函数对原信号的每一帧进行变换，应该是为了消除关联性
    参数定义：
    frames:audio2frame函数返回的帧矩阵
    signal_length:信号长度
    frame_length:帧长度
    frame_step:帧间隔
    winfunc:对每一帧加window函数进行分析，默认此处不加window
    '''
    #对参数进行取整操作
    signal_length=round(signal_length) #信号的长度
    frame_length=round(frame_length) #帧的长度
    frames_num=numpy.shape(frames)[0] #帧的总数
    assert numpy.shape(frames)[1]==frame_length,'"frames"矩阵大小不正确，它的列数应该等于一帧长度'  #判断frames维度 
    indices=numpy.tile(numpy.arange(0,frame_length),(frames_num,1))+numpy.tile(numpy.arange(0,frames_num*frame_step,frame_step),(frame_length,1)).T  #相当于对所有帧的时间点进行抽取，得到frames_num*frame_length长度的矩阵
    indices=numpy.array(indices,dtype=numpy.int32)
    pad_length=(frames_num-1)*frame_step+frame_length #铺平后的所有信号
    if signal_length<=0:
        signal_length=pad_length
    recalc_signal=numpy.zeros((pad_length,)) #调整后的信号
    window_correction=numpy.zeros((pad_length,1)) #窗关联
    win=winfunc(frame_length)
    for i in range(0,frames_num):
        window_correction[indices[i,:]]=window_correction[indices[i,:]]+win+1e-15 #表示信号的重叠程度
        recalc_signal[indices[i,:]]=recalc_signal[indices[i,:]]+frames[i,:] #原信号加上重叠程度构成调整后的信号
    recalc_signal=recalc_signal/window_correction #新的调整后的信号等于调整信号处以每处的重叠程度 
    return recalc_signal[0:signal_length] #返回该新的调整信号

def spectrum_magnitude(frames,NFFT):
    '''计算每一帧经过FFY变幻以后的频谱的幅度，若frames的大小为N*L,则返回矩阵的大小为N*NFFT
    参数说明：
    frames:即audio2frame函数中的返回值矩阵，帧矩阵
    NFFT:FFT变换的数组大小,如果帧长度小于NFFT，则帧的其余部分用0填充铺满
    '''
    complex_spectrum=numpy.fft.rfft(frames,NFFT) #对frames进行FFT变换
    return numpy.absolute(complex_spectrum)  #返回频谱的幅度值
    
def spectrum_power(frames,NFFT):
    '''计算每一帧傅立叶变换以后的功率谱
    参数说明：
    frames:audio2frame函数计算出来的帧矩阵
    NFFT:FFT的大小
    '''
    return 1.0/NFFT * numpy.square(spectrum_magnitude(frames,NFFT)) #功率谱等于每一点的幅度平方/NFFT

def log_spectrum_power(frames,NFFT,norm=1):
    '''计算每一帧的功率谱的对数形式
    参数说明：
    frames:帧矩阵，即audio2frame返回的矩阵
    NFFT：FFT变换的大小
    norm:范数，即归一化系数
    '''
    spec_power=spectrum_power(frames,NFFT)
    spec_power[spec_power<1e-30]=1e-30 #为了防止出现功率谱等于0，因为0无法取对数
    log_spec_power=10*numpy.log10(spec_power)
    if norm:
        return log_spec_power-numpy.max(log_spec_power)
    else:
        return log_spec_power

def pre_emphasis(signal,coefficient=0.95):
    '''对信号进行预加重
    参数含义：
    signal:原始信号
    coefficient:加重系数，默认为0.95
    '''
    return numpy.append(signal[0],signal[1:]-coefficient*signal[:-1])


# ----------------------------------------------------------------------------------------------------------

#coding=utf-8
# 计算每一帧的MFCC系数
# 张泽旺，2015-12-13

import numpy
# from sigprocess import audio2frame
# from sigprocess import pre_emphasis
# from sigprocess import spectrum_power
from scipy.fftpack import dct 
#首先，为了适配版本3.x，需要调整xrange的使用，因为对于版本2.x只能使用range，需要将xrange替换为range
try:
    xrange(1)
except:
    xrange=range


def calcMFCC_delta_delta(signal,samplerate=16000,win_length=0.025,win_step=0.01,cep_num=13,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,cep_lifter=22,appendEnergy=True):
    '''计算13个MFCC+13个一阶微分系数+13个加速系数,一共39个系数
    '''
    feat=calcMFCC(signal,samplerate,win_length,win_step,cep_num,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff,cep_lifter,appendEnergy)   #首先获取13个一般MFCC系数
    result1=derivate(feat)
    result2=derivate(result1)
    result3=numpy.concatenate((feat,result1),axis=1)
    result=numpy.concatenate((result3,result2),axis=1)
    return result


def calcMFCC_delta(signal,samplerate=16000,win_length=0.025,win_step=0.01,cep_num=13,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,cep_lifter=22,appendEnergy=True):
    '''计算13个MFCC+13个一阶微分系数
    '''
    feat=calcMFCC(signal,samplerate,win_length,win_step,cep_num,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff,cep_lifter,appendEnergy)   #首先获取13个一般MFCC系数
    result=derivate(feat) #调用derivate函数
    result=numpy.concatenate((feat,result),axis=1)
    return result    
     
def derivate(feat,big_theta=2,cep_num=13):
    '''计算一阶系数或者加速系数的一般变换公式
    参数说明:
    feat:MFCC数组或者一阶系数数组
    big_theta:公式中的大theta，默认取2
    '''
    result=numpy.zeros(feat.shape) #结果
    denominator=0  #分母
    for theta in numpy.linspace(1,big_theta,big_theta):
        denominator=denominator+theta**2
    denominator=denominator*2 #计算得到分母的值
    for row in numpy.linspace(0,feat.shape[0]-1,feat.shape[0]):
        tmp=numpy.zeros((cep_num,))
        numerator=numpy.zeros((cep_num,)) #分子
        for t in numpy.linspace(1,cep_num,cep_num):
            a=0
            b=0
            s=0
            for theta in numpy.linspace(1,big_theta,big_theta):
                if (t+theta)>cep_num:
                    a=0
                else:
                    a=feat[int(row)][int(t+theta-1)]
                if (t-theta)<1:
                    b=0
                else:
                    b=feat[int(row)][int(t-theta-1)]
                s+=theta*(a-b)
            numerator[int(t-1)]=s
        tmp=numerator*1.0/denominator
        result[int(row)]=tmp
    return result  


def calcMFCC(signal,samplerate=16000,win_length=0.025,win_step=0.01,cep_num=13,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,cep_lifter=22,appendEnergy=True):
    '''计算13个MFCC系数
    参数含义：
    signal:原始音频信号，一般为.wav格式文件
    samplerate:抽样频率，这里默认为16KHz
    win_length:窗长度，默认即一帧为25ms
    win_step:窗间隔，默认情况下即相邻帧开始时刻之间相隔10ms
    cep_num:倒谱系数的个数，默认为13
    filters_num:滤波器的个数，默认为26
    NFFT:傅立叶变换大小，默认为512
    low_freq:最低频率，默认为0
    high_freq:最高频率
    pre_emphasis_coeff:预加重系数，默认为0.97
    cep_lifter:倒谱的升个数？？
    appendEnergy:是否加上能量，默认加
    '''
    
    feat,energy=fbank(signal,samplerate,win_length,win_step,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff)
    feat=numpy.log(feat)
    feat=dct(feat,type=2,axis=1,norm='ortho')[:,:cep_num]  #进行离散余弦变换,只取前13个系数
    feat=lifter(feat,cep_lifter)
    if appendEnergy:
        feat[:,0]=numpy.log(energy)  #只取2-13个系数，第一个用能量的对数来代替
    return feat

def fbank(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97):
    '''计算音频信号的MFCC
    参数说明：
    samplerate:采样频率
    win_length:窗长度
    win_step:窗间隔
    filters_num:梅尔滤波器个数
    NFFT:FFT大小
    low_freq:最低频率
    high_freq:最高频率
    pre_emphasis_coeff:预加重系数
    '''
    
    high_freq=high_freq or samplerate/2  #计算音频样本的最大频率
    signal=pre_emphasis(signal,pre_emphasis_coeff)  #对原始信号进行预加重处理
    frames=audio2frame(signal,win_length*samplerate,win_step*samplerate) #得到帧数组
    spec_power=spectrum_power(frames,NFFT)  #得到每一帧FFT以后的能量谱
    energy=numpy.sum(spec_power,1)  #对每一帧的能量谱进行求和
    energy=numpy.where(energy==0,numpy.finfo(float).eps,energy)  #对能量为0的地方调整为eps，这样便于进行对数处理
    fb=get_filter_banks(filters_num,NFFT,samplerate,low_freq,high_freq)  #获得每一个滤波器的频率宽度
    feat=numpy.dot(spec_power,fb.T)  #对滤波器和能量谱进行点乘
    feat=numpy.where(feat==0,numpy.finfo(float).eps,feat)  #同样不能出现0
    return feat,energy
   
def log_fbank(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97):
    '''计算对数值
    参数含义：同上
    '''
    feat,energy=fbank(signal,samplerate,win_length,win_step,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff)
    return numpy.log(feat)

def ssc(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97):
    '''
    待补充
    ''' 
    high_freq=high_freq or samplerate/2
    signal=sigprocess.pre_emphasis(signal,pre_emphasis_coeff)
    frames=sigprocess.audio2frame(signal,win_length*samplerate,win_step*samplerate)
    spec_power=sigprocess.spectrum_power(frames,NFFT) 
    spec_power=numpy.where(spec_power==0,numpy.finfo(float).eps,spec_power) #能量谱
    fb=get_filter_banks(filters_num,NFFT,samplerate,low_freq,high_freq) 
    feat=numpy.dot(spec_power,fb.T)  #计算能量
    R=numpy.tile(numpy.linspace(1,samplerate/2,numpy.size(spec_power,1)),(numpy.size(spec_power,0),1))
    return numpy.dot(spec_power*R,fb.T)/feat

def hz2mel(hz):
    '''把频率hz转化为梅尔频率
    参数说明：
    hz:频率
    '''
    return 2595*numpy.log10(1+hz/700.0)

def mel2hz(mel):
    '''把梅尔频率转化为hz
    参数说明：
    mel:梅尔频率
    '''
    return 700*(10**(mel/2595.0)-1)

def get_filter_banks(filters_num=20,NFFT=512,samplerate=16000,low_freq=0,high_freq=None):
    '''计算梅尔三角间距滤波器，该滤波器在第一个频率和第三个频率处为0，在第二个频率处为1
    参数说明：
    filers_num:滤波器个数
    NFFT:FFT大小
    samplerate:采样频率
    low_freq:最低频率
    high_freq:最高频率
    '''
    #首先，将频率hz转化为梅尔频率，因为人耳分辨声音的大小与频率并非线性正比，所以化为梅尔频率再线性分隔
    low_mel=hz2mel(low_freq)
    high_mel=hz2mel(high_freq)
    #需要在low_mel和high_mel之间等间距插入filters_num个点，一共filters_num+2个点
    mel_points=numpy.linspace(low_mel,high_mel,filters_num+2)
    #再将梅尔频率转化为hz频率，并且找到对应的hz位置
    hz_points=mel2hz(mel_points)
    #我们现在需要知道这些hz_points对应到fft中的位置
    bin=numpy.floor((NFFT+1)*hz_points/samplerate)
    #接下来建立滤波器的表达式了，每个滤波器在第一个点处和第三个点处均为0，中间为三角形形状
    fbank=numpy.zeros([filters_num, int(NFFT/2+1)])
    for j in xrange(0,filters_num):
        for i in xrange(int(bin[j]),int(bin[j+1])):
            fbank[j,i]=(i-bin[j])/(bin[j+1]-bin[j])
        for i in xrange(int(bin[j+1]),int(bin[j+2])):
            fbank[j,i]=(bin[j+2]-i)/(bin[j+2]-bin[j+1])
    return fbank

def lifter(cepstra,L=22):
    '''升倒谱函数
    参数说明：
    cepstra:MFCC系数
    L：升系数，默认为22
    '''
    if L>0:
        nframes,ncoeff=numpy.shape(cepstra)
        n=numpy.arange(ncoeff)
        lift=1+(L/2)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        return cepstra


# ----------------------------------------------------------------------------------------------------

import scipy.io.wavfile as wav
import numpy

(rate,sig) = wav.read("music/mu1.wav")
print(len(sig), rate)
# mipa
mfcc_feat = calcMFCC_delta_delta(signal = sig, samplerate = rate, win_length = len(sig)/rate) 
print(mfcc_feat.shape)