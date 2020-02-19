import numpy as np
import random



def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

if __name__ == "__main__":
	file_all = open("/home/daniel/Documents/LU_Net_All_Labels_Mapped/LU-Net/lunet/data/semantic_all.txt", "r")
	instances = 23201
	val_split = 0.2 #(20%)

	#validation instance = 1, train instance = 0
	num_val = np.round(val_split*instances)
	n = np.zeros(instances)
	for i in range (num_val.astype(int)): 
		r = random.randrange(0,instances,1)
		if (n[r]==1):
			i -=1
		else:
			n[r]=1
	
	filet = open("train.txt","w")
	filev = open("val.txt","w")


	for i in range (instances):
		line = file_all.readline()
		if(n[i]==1):
			filev.write(line)
		else:
			filet.write(line)

	file_all.close()
	filet.close()
	filev.close()
