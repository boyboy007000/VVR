import os
path = os.path.dirname(os.path.abspath(__file__))
for fname in os.listdir(path):
	if fname[0] == 's':
		num = int(fname[1:len(fname)-4])
		snum = fname[1:len(fname)-4]
		snum2 = snum.zfill(4)
		os.rename(os.path.join(path,fname),os.path.join(path,'s'+snum2+'.wav'))

