import numpy as np
f = open('main.csv', 'rb').read()
seq_len=258
data = f.decode().split('\n')
#print(data)
sequence_length = seq_len - 1
result = []
for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])
result = np.array(result)
row = round(0.9 * result.shape[0])
train = result[:int(row)]
np.random.shuffle(train)
x_train = train[:-1]
y_train = train[-1]
x_test = result[int(row):, :-1]
y_test = result[int(row):, -1]