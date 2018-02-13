import os

phuam =   ['ngh','qu','ch','gh','kh','ng','nh','ph','th','tr','gi','dd','b','c','d','g','h','k','l','m','n','p','r','s','t','v','x']
sophuam = [  0  , 17 , 3  , 5  , 7  , 1  , 8  , 9  , 10 , 4  , 11 , 14 , 15, 16, 12, 6 , 2 , 18, 19, 20, 21, 22, 13,23 , 25, 26,24 ]
nguyenam =   ['u+' ,'o^','o+','e^','a(','a^','a','e','i','o','u','y']
songuyenam = [  0  ,  1 ,  2 , 3  , 4  , 5  , 6 , 7 , 8 , 9 , 10, 11]
dau =   ['`',"'",'.','~','?']
sodau = [ 0 , 1 , 2 , 3 , 4 ]


#kiem tra chi so lien tuc
for i in range(max(sophuam)):
    if i not in sophuam:
        print("sophuam khong lien tuc")
        exit()
for i in range(max(songuyenam)):
    if i not in songuyenam:
        print("songuyenam khong lien tuc")
        exit()
for i in range(max(sodau)):
    if i not in sodau:
        print("sodau khong lien tuc")
        exit()
        
f = open(os.path.dirname(os.path.abspath(__file__))+"\\transcript_viqr.txt",'r')
s = f.readline()
w = s.split()
for i in range(len(w)):
    #tim phu am
    for snd in phuam:
        if w[i].startswith(snd):
            print("phu am: " + snd)
            w[i] = w[i][len(snd):]
    #tim dau
    for snd in dau:
        pos = w[i].find(snd)
        if pos > 0:
            print("dau: " + snd)
            w[i] = w[i][:pos]+w[i][pos+1:]
            print(w[i])
