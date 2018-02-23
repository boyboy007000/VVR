import wave
import struct
import random
import os
import sys

numframe3s = 24000
scale = 0.997
minrnd = round(-32768*(1-scale)) + 1
maxrnd = round(32767 *(1-scale)) - 1

fl = os.listdir(os.path.dirname(os.path.abspath(__file__)))
for f in fl:
    if f.startswith('s') and f.endswith('wav'):
        w = wave.open(os.path.dirname(os.path.abspath(__file__))+ "\\" + f,"rb")
        w.rewind();
        if w.getnframes() > numframe3s:
            print(f + " more than 3s")
        elif w.getnframes() < numframe3s:
            ww = wave.open(os.path.dirname(os.path.abspath(__file__))+ "\\x" + f[1:],"wb")
            pr = w.getparams()
            ons = pr.nframes;
            pr = pr._replace(nframes=numframe3s)
            ww.setparams(pr)
            for i in range(0,ons):
                ww.writeframesraw(struct.pack("<h",round(struct.unpack("<h",w.readframes(1))[0]*scale)+random.randint(minrnd,maxrnd)))
            for i in range(ons,numframe3s):
                ww.writeframesraw(struct.pack("<h",random.randint(minrnd,maxrnd)))
            ww.close()
        w.close()            
            
