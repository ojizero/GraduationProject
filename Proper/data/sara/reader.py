import serial.tools.list_ports
import platform

def get_ports (name='generic'):
	# get all serial ports with device `name`
	serial_ports = [port for port in serial.tools.list_ports.comports() if name.lower() in port[1].lower()]

	if len(serial_ports) == 0:
		raise EnvironmentError('couldn\'t find any ports')

	return serial_ports

# def get_ports():
# 	# Not sure how OSX displays the name of the UNO port
# 	device_name = {'Darwin': 'arduino', 'Linux': 'ACM'}
# 	serial_ports = [port for port in serial.tools.list_ports.comports() if device_name[platform.system()] in port[1].lower()]
# 	if len(serial_ports) == 0:
# 		raise EnvironmentError('Couldn\'t find any ports')
# 	return serial_ports

if __name__ == '__main__':
	import os
	import serial
	import datetime
	from sys import argv

	if len(argv) < 3:
		raise ValueError('missing script arguments')

	letter = argv[1]
	readings_count = int(argv[2])

	# serial_port = get_feather_ports()[0][0]
	serial_port = get_ports('ACM')[0][0]
	s = serial.Serial(serial_port, 115200)

	if not os.path.exists('./%s' % letter):
	    os.makedirs('./%s' % letter)

	for _ in range(readings_count):
		c = datetime.datetime.now()
		c = '%s_%s_%s_%s_%s' % (c.month, c.day, c.hour, c.minute, c.second)
		input('\n\ncharacter ')
		s.write(b'r')
		with open('./%s/%s.%s.csv' % (letter, letter, c), 'wb') as f:
			while True:
				try:
					l = s.readline()
					# if 'failed' in l.decode():
					# 	raise Exception(l.decode())

					f.write(l)
					print(l.decode())
				except KeyboardInterrupt:
					s.write(b's')
					break
				except Exception as e:
					print('-=-=-=-=-=-= errored =-=-=-=-=-=-', e)
					exit(1)
