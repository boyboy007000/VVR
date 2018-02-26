import os
from wavelet_gen import nninput
from parse import nnoutput

tf = open(os.path.dirname(os.path.abspath(__file__))+ "\\traindata.dat", "wb")
dout = nnoutput()
din = nninput(1)
numinp = 0
for co in din:
    numinp = numinp + len(co)

#for i in range(1,len(dout)+1):
 #   co = nninput(n)
    
