import serial

s = serial.Serial('/dev/ttyACM0', 115200)

s.write(b'r')

with open('/home/oji/echoed', 'wb') as f:
	while True:
		try:
			f.write(s.readline())
		except Exception as e:
			pass

