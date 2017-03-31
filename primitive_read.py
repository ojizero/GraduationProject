import serial
from time import clock as now

s = serial.Serial('/dev/ttyACM0', 115200)

s.write(b'r')

with open('./echoed', 'wb') as f:
	while True:
		try:
			l = s.readline()
			f.write(l)
			print(now(), l.decode())
		except Exception as e:
			print('errored', e)
			break

