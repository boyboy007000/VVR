import wave
import struct
import pywt
import matplotlib.pyplot as plt
import os
import sys

fl = os.listdir(os.path.dirname(os.path.abspath(__file__)))
for f in fl:
    if f.startswith('s'):
        w = wave.open(os.path.dirname(os.path.abspath(__file__))+ "\\" + f,"rb")
        w.rewind();
        print(w.getnframes())
        w.close()
