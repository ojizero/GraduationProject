import serial.tools.list_ports

def get_feather_ports ():
	# get all serial ports with device 'Feather'
	serial_ports = [port for port in serial.tools.list_ports.comports() if 'feather' in port[1].lower()]

	if len(serial_ports) == 0:
		raise EnvironmentError('couldn\'t find any ports')

	return serial_ports


if __name__ == '__main__':
	import os
	import serial
	from sys import argv
	from time import clock as now

	if len(argv) < 3:
		raise ValueError('missing script arguments')

	letter = argv[1]
	readings_count = int(argv[2])

	serial_port = get_feather_ports()[0][0]
	s = serial.Serial(serial_port, 115200)

	if not os.path.exists('./%s' % letter):
	    os.makedirs('./%s' % letter)

	s.write(b'r')
	for _ in range(readings_count):
		with open('./%s/%s.%s.csv' % (letter, letter, now()), 'wb') as f:
			while True:
				try:
					l = s.readline()
					f.write(l)
					print(now(), l.decode())
				except KeyboardInterrupt:
					break
				except Exception as e:
					print('-=-=-=-=-=-= errored =-=-=-=-=-=-', e)
					break

