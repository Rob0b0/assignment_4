# CSS584

训练集位置：./train/music/*.wav 和 ./train/speech/*.wav <br />
测试集位置：./test/music/*.wav 和 ./test/speech/*.wav <br />
默认音频： ./Mizuki_Nana.wav <br />
<hr />
<p>
<strong>2018-11-30 03:57:00</strong> <br />
本次更新 <br />
· 增加了决策树模型 <br />
· 调整了GUI <br />
· 为乱点狂魔设置了程序状态的判定，妈妈再也不用担心我被人开一百个线程辣 <br />
· 其他无关痛痒的小调整 <br />
潜在bug： <br />
· 先训练模型，之后用new test更改测试集为某较长的wav文件（比如./Mizuki_Nana.wav）再进行predict。此过程中可以建立多个predict线程。原因不明。 <br />
· easygui的文件打开框，文件类型指定格式不正确，无法有效限定为wav文件。在new test过程中可能引入非wav类型的文件，造成未知错误。 <br />
</p>
<br />
<p>
<strong>2018-11-28 07:24:00</strong> <br />
本次更新 <br />
· 修正了PCA过程中的错误 <br />
· 修改了训练集路径 <br />
· 为自由选取测试集做好数据结构上的准备（label：‘UNKNOW’） <br />
· 试图增加MFCC然而并不成功，姑且放上去了，但是引用处被注释了 <br />
· 其他无关痛痒的小调整 <br />
</p>
<hr />
缺属性值啊啊啊啊啊啊啊 <br />
频率的横坐标可能不太对（什么叫可能，就是不太对 <br />
一个文件就是一段音频，没有进一步的分割（会声会影大法好，退pyAudioAnalysis保平安，搜五评Python3，有真相（x <br />
label是根据文件直属文件夹的名字自动归类的，所以还没想好怎么通过gui选取文件训练集和测试集（划掉）能选个测试集就完事儿了还要啥自行车儿！ <br />
<hr />
自从换上了白学音乐大全和6分30秒White Album，腰不酸了腿不疼了，KNN准确率和召回率突破90%不费劲！ <br />
但是SVM哭给你看！ <br />
（RNN：那我呢？ <br />
好吧并没有什么琴梨用 <br />

