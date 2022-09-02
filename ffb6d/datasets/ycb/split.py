import numpy as np
file1 = open('data.txt', 'r')
Lines = file1.readlines()
choice = np.random.choice(range(len(Lines)), size=(113845,), replace=False)    
ind = np.zeros(len(Lines), dtype=bool)
ind[choice] = True
rest = ~ind
train = open('train_all.txt', 'w')
test = open('test_all.txt', 'w')
for i in range(len(Lines)):
    if rest[i] == True:
        train.writelines(Lines[i])
    else:
        test.writelines(Lines[i])