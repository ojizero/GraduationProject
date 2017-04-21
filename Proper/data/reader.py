import serial.tools.list_ports
import platform
def get_feather_ports ():
	# get all serial ports with device 'Feather'
	serial_ports = [port for port in serial.tools.list_ports.comports() if 'feather' in port[1].lower()]

	if len(serial_ports) == 0:
		raise EnvironmentError('couldn\'t find any ports')

	return serial_ports

def get_ports():
	# Not sure how OSX displays the name of the UNO port
	device_name = {'Darwin': 'arduino', 'Linux': 'ACM'}
	serial_ports = [port for port in serial.tools.list_ports.comports() if device_name[platform.system()] in port[1].lower()]
	if len(serial_ports) == 0:
		raise EnvironmentError('Couldn\'t find any ports')
	return serial_ports

if __name__ == '__main__':
	import os
	import serial
	from sys import argv
	from time import sleep, time as now
	import datetime

	if len(argv) < 3:
		raise ValueError('missing script arguments')

	letter = argv[1]
	readings_count = int(argv[2])

	# serial_port = get_feather_ports()[0][0]
	serial_port = get_ports()[0][0]
	s = serial.Serial(serial_port, 115200)

	if not os.path.exists('./%s' % letter):
	    os.makedirs('./%s' % letter)

	input('pausing')
	s.write(b'r')
	for _ in range(readings_count):
		c = datetime.datetime.now()
		c = '%s_%s_%s_%s_%s' % (c.month, c.day, c.hour, c.minute, c.second)
		with open('./%s/%s.%s.csv' % (letter, letter, c), 'wb') as f:
			while True:
				try:
					l = s.readline()
					f.write(l)
					print(now(), l.decode())
				except KeyboardInterrupt:
					# sleep(1)
					input('pausing')
					break
				except Exception as e:
					print('-=-=-=-=-=-= errored =-=-=-=-=-=-', e)
					break
