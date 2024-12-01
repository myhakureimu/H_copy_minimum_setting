import pickle


with open('strings.pkl', 'rb') as f:
    strings = pickle.load(f)

for i in range(12):
    for j in range(2):
        key1 = list(strings['train'].keys())[i]
        key2 = list(strings['train'][key1].keys())[j]
        #print(key1)
        #print(key2)
        print(len(strings['train'][key1][key2]), len(strings['test_'][key1][key2]))