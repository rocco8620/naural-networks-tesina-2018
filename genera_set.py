import json,sys

INPUT_FILE = "dati_fixati.txt"
OUTPUT_FILE_QUESTIONS = "out_x.txt"
OUTPUT_FILE_RESULTS = "out_y.txt"

DATASET_LEN = 720

data = []

def avg(dat):
	acc = 0
	for d in dat:
		acc += d
	return acc/len(dat)	

def determine(values):
	deviation = 0.004
	ma = max(values)
	mi = min(values)

	diff_max = abs(values[0]-ma)
	diff_min = abs(values[0]-mi)

	if diff_max > diff_min:
		if ma > values[0]*(1+deviation): # sale
			return 0
		else: # nulla
			return 2	
	else:
		if mi < values[0]*(1-deviation): # scende
			return 1
		else: # nulla
			return 2

	
	
		
		
with open(INPUT_FILE, 'r') as f:
	for line in f:
		pieces = line.split(':')
		#data.append([float(pieces[1]),float(pieces[2]),float(pieces[3]),float(pieces[4])])
		data.append([float(pieces[1])])

array_x = []
array_y = []



parsed_1 = []

i = 0
data_len = int(len(data)//DATASET_LEN)/1000
while i < int(len(data)/DATASET_LEN):
	parsed_1.append(data[i*DATASET_LEN : (i+1)*DATASET_LEN])
	#print(parsed_1[0])
	#print(len(parsed_1[0]))
	#print(len(parsed_1))	
	#sys.exit()
	if i % 1000 == 0: print("F1",i//1000,"/",data_len)
	i += 1



i = 0
data_len = (len(parsed_1)/1000)+1
while i < len(parsed_1)-1:
	flat = [item for sublist in parsed_1[i] for item in sublist]
	array_x.append(flat)
	tmp = [0,0,0]
	flat = [ item[0] for item in parsed_1[i+1] ] 
	#print(len(flat))
	#sys.exit()
	tmp[ determine(flat) ] = 1
	array_y.append(tmp)

	i+= 1
	if i % 1000	 == 0: print("F2",i/1000,"/",data_len)


i0 = 0
i1 = 0
i2 = 0

for x in array_y:
	i0 += x[0]
	i1 += x[1]
	i2 += x[2]

print(i0,i1,i2)
#sys.exit()

out_y = open(OUTPUT_FILE_RESULTS, 'w')
#array_y = [item for sublist in array_y for item in sublist]
print(len(array_y))
print(len(array_y[0]))
json.dump(array_y,out_y)

out_x = open(OUTPUT_FILE_QUESTIONS, 'w')
#array_x = [item for sublist in array_x for item in sublist]
print(len(array_x))
print(len(array_x[0]))
json.dump(array_x,out_x)