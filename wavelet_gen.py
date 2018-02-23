import wave
import struct
import pywt
import matplotlib.pyplot as plt
import os

w = wave.open(os.path.dirname(os.path.abspath(__file__))+ "\\s0001.wav","rb")
w.rewind()
p = w.getparams()
print(p)
p = p._replace(nframes=20)
print(p)

exit()
data = []
for i in range(0,w.getnframes()):
    data.append(struct.unpack("<h",w.readframes(1))[0])
db1 = pywt.Wavelet('db1')
co = pywt.wavedec(data,db1)
w.close()
