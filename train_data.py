import os
import struct
from wavelet_gen import nninput
from parse import nnoutput

tf = open(os.path.dirname(os.path.abspath(__file__))+ "\\traindata.dat", "wb")
dout = nnoutput()
for i in range(0,len(dout)):
    din = nninput(i+1)
    numinp = 0
    numout = 0
    for s in dout:
        if len(s) > numout:
            numout = len(s)
    for co in din:
        numinp = numinp + len(co)
    tf.write(struct.pack("<I",len(dout)*2))
    tf.write(struct.pack("<I",numinp+1))
    tf.write(struct.pack("<I",numout))
    #write input
    for co in din:
        for x in co:
            tf.write(struct.pack("<d",x))
    tf.write(struct.pack("<d",1.0))
    #write output
    for x in dout[i]:
        tf.write(struct.pack("<d",x))
    if numout > len(dout[i]):
        for j in range(len(dout[i]),numout):
            tf.write(struct.pack("<d",-0.9))
    #write input 2
    for co in din:
        for x in co:
            tf.write(struct.pack("<d",x))
    tf.write(struct.pack("<d",-1.0))
    #write output 2
    if numout > len(dout[i]):
        for j in range(len(dout[i]),numout):
            tf.write(struct.pack("<d",-0.9))
    for x in reversed(dout[i]):
        tf.write(struct.pack("<d",x))
tf.close()
#for i in range(1,len(dout)+1):
 #   co = nninput(n)
    
