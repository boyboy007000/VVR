import os
import struct
from wavelet_gen import nninput
from parse import nnoutput

tf = open(os.path.dirname(os.path.abspath(__file__))+ "\\traindata.dat", "wb")
dout = nnoutput()
din = nninput(1)
numinp = 0
numout = 0
for s in dout:
    if len(s) > numout:
        numout = len(s)
for co in din:
    numinp = numinp + len(co)
tf.write(struct.pack("<I",len(dout)))
tf.write(struct.pack("<I",numinp))
tf.write(struct.pack("<I",numout+1))
for co in din:
    for x in co:
        tf.write(struct.pack("<d",x))

tf.close()
#for i in range(1,len(dout)+1):
 #   co = nninput(n)
    
