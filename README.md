### 如何开始?

```python
pip install -r ./requirements.txt
# 注意需要安装 ffmpeg
# 如果在安装pyaudio的时候出现了故障，请检查是否缺失了 portaudio 如果是请先安装这个包
```


### 文件说明

* `data`: 是用来存放和弦手势图或者其他数据的
* `chord_recognize.py`: 是和弦手势识别的一个例子
* `chormgram_recognize.py`: 是利用色谱图识别各个音所占比重的例子
* `pitch_recognizer.py`: 是一个识别所有音高的例子
* `recognizer.py`: 是一个手势识别器，识别吉他手势的
