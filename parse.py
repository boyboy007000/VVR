import os

snd =   ['ngh','qu','ch','gh','kh','ng','nh','ph','th','tr','gi','dd','b','c','d','g','h','k','l','m','n','p','r','s','t','v','x','u+','o^','o+','e^','a(','a^','a','e','i','o','u','y','`',"'",'.','~','?']
sndi =  [  0  , 17 , 3  , 5  , 7  , 1  , 8  , 9  , 10 , 4  , 11 , 14 , 15, 16, 12, 6 , 2 , 18, 19, 20, 21, 22, 13,23 , 25, 26,24 , 27 , 28 , 29 , 30 , 31 , 32 , 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]

def nnoutput()
    #kiem tra chi so lien tuc
    for i in range(max(sndi)):
        if i not in sndi:
            print("sndi khong lien tuc")
            exit()
    data = []
    f = open(os.path.dirname(os.path.abspath(__file__))+"\\transcript_viqr.txt",'r')
    maxns = 0
    maxline = 0
    maxsndi = max(sndi)
    interval = 1.8 / len(snd)
    j = 0
    for line in f:
        j = j + 1
        w = line.split()
        ns = len(w)
        linedata = []
        for i in range(len(w)):
            while len(w[i]) > 0:
                found = 0
                for ind,sn in enumerate(snd):
                    if w[i].startswith(sn):
                        found = 1
                        w[i] = w[i][len(sn):]
                        ns = ns + 1
                        linedata.append(-0.9+(sndi[ind]+1)*interval)
                        break
                if not found:
                    print(line + ":" + i)
                    exit()
            linedata.append(-0.9) #end of word
        data.append(linedata)
    return data
