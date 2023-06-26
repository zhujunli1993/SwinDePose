import numpy as np

file_lst = np.loadtxt('/workspace/DATA/Linemod_preprocessed/renders_nrm/phone/file_list_full.txt', dtype=str)
file_short = []
ind = np.random.randint(0, len(file_lst), 10000)

for i in ind:
    file_short.append(file_lst[i])
with open('/workspace/DATA/Linemod_preprocessed/renders_nrm/phone/file_list.txt', 'w') as f:
    for line in file_short:
        f.write(line)
        f.write('\n')
