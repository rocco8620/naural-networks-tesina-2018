INPUT_FILE  = "EURUSD_00-17.txt"
OUTPUT_FILE = "dati_fixati.txt"


out = open(OUTPUT_FILE, "w")

import datetime
import time
import sys

base = datetime.datetime.strptime("01-01-2000 00:00", "%d-%m-%Y %H:%M")
base = time.mktime(base.timetuple())

i = 0
data = []
print("CHECK 1")
with open(INPUT_FILE, 'r') as f:
	for line in f:
		pieces = line.split(';')
		date_time = pieces[0][6:8]+"-"+pieces[0][4:6]+"-"+pieces[0][0:4]+" "+pieces[0][9:11]+":"+pieces[0][11:13]
		open_value  = float(pieces[1])
		#max_value   = float(pieces[2])
		#min_value   = float(pieces[3])
		#close_value = float(pieces[4])
		d = datetime.datetime.strptime(date_time, "%d-%m-%Y %H:%M")
		d = time.mktime(d.timetuple())
		d = int(int(d-base) / 60)

		#data.append((d,open_value,max_value,min_value,close_value))
		data.append((d,open_value))

		if i % 10000 == 0: print(i/10000,"/ x")
		i += 1


print("CHECK 2")
val = 1
i = 0


while i < len(data)-1:
	#print("CHECK 3")
	if data[i][0]+1 != data[i+1][0]:
		# se i valori non sono continuativi
		if data[i+1][0] - data[i][0] < 1024:
			# se ci troviamo in un weekend facciamo finta di nulla
			j = data[i+1][0] - data[i][0] - 1
			print("Riempio un gap di",j)
			while j > 0:
				#out.write(str(val)+':'+str(data[i][1])+':'+str(data[i][2])+':'+str(data[i][3])+':'+str(data[i][4])+'\r\n')

				out.write(str(val)+':'+str(data[i][1])+'\r\n')
				val += 1
				j -= 1
	else:	
		if data[i+1][0] - data[i][0] > 1: print("Gap di",data[i+1][0] - data[i][0],"ignorato")	
		# se i valori sono continuativi
		#out.write(str(val)+':'+str(data[i][1])+':'+str(data[i][2])+':'+str(data[i][3])+':'+str(data[i][4])+'\r\n')
		out.write(str(val)+':'+str(data[i][1])+'\r\n')
		val += 1
		
	i+=1
	if i % 10000 == 0: print("I:",i,"VAL:",val)

out.close()	

