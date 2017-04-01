import serial
from time import clock as now

serial_port = '/dev/cu.usbmodem1411' # '/dev/ttyACM0' # for linux

s = serial.Serial(serial_port, 115200)

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

