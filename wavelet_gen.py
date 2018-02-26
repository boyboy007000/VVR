import wave
import struct
import pywt
import matplotlib.pyplot as plt
import os

def nninput(n):
    w = wave.open(os.path.dirname(os.path.abspath(__file__))+ "\\x"+str(n).zfill(4)+".wav","rb")
    w.rewind()
    data = []
    for i in range(0,w.getnframes()):
        data.append(struct.unpack("<h",w.readframes(1))[0])
    w.close()
    db1 = pywt.Wavelet('db2')
    co = pywt.wavedec(data,db1,level=6)
    maxn = 0
    for i in range(0,len(co)):
        m = max(abs(co[i]))
        if m > maxn:
            maxn = m
#    print(maxn)
    for i in range(0,len(co)):
        co[i] = co[i] / maxn
#    print(len(rec))
#    plt.plot(rec)
#    plt.show()
    return co
