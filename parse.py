import os

snd =   ['ngh','qu','ch','gh','kh','ng','nh','ph','th','tr','gi','dd','b','c','d','g','h','k','l','m','n','p','r','s','t','v','x','u+','o^','o+','e^','a(','a^','a','e','i','o','u','y','`',"'",'.','~','?']
sndi =  [  0  , 17 , 3  , 5  , 7  , 1  , 8  , 9  , 10 , 4  , 11 , 14 , 15, 16, 12, 6 , 2 , 18, 19, 20, 21, 22, 13,23 , 25, 26,24 , 27 , 28 , 29 , 30 , 31 , 32 , 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]

#kiem tra chi so lien tuc
for i in range(max(sndi)):
    if i not in sndi:
        print("sndi khong lien tuc")
        exit()
        
f = open(os.path.dirname(os.path.abspath(__file__))+"\\transcript_viqr.txt",'r')
for line in f:
    w = line.split()
    for i in range(len(w)):
        while len(w[i]) > 0:
            for sn in snd:
                if w[i].startswith(sn):
                    print(sn + " ", end = '')
                    w[i] = w[i][len(sn):]
                    break
        print("")
