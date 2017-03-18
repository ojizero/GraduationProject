#include <Wire.h>
#include <SPI.h>
#include <LSM9DS1.h>

#include "Navio/Util.h"
#include "Navio/AHRS.hpp"


LSM9DS1 imu;
AHRS ahrs;

int print_flag = 1;
long current, previous, dt;

#define LSM9DS1_M 0x1E // Would be 0x1C if SDO_M is LOW
#define LSM9DS1_AG  0x6B // Would be 0x6A if SDO_AG is LOW

#define PRINT_CALCULATED
// #define PRINT_RAW
#define PRINT_SPEED 250 // 250 ms between prints

#define DECLINATION -8.58 // Declination (degrees) in Boulder, CO.

#define TCAADDR 0x70
void tcaselect(uint8_t i)
{
	if (i > 7) return;

	Wire.beginTransmission(TCAADDR);
	Wire.write(1 << i);
	Wire.endTransmission();
}

void setup()
{
	int select_line = 0;

	Wire.begin();
	Serial.begin(115200);

	// Before initializing the IMU, there are a few settings
	// we may need to adjust. Use the settings struct to set
	// the device's communication mode and addresses:
	imu.settings.device.commInterface = IMU_MODE_I2C;
	imu.settings.device.mAddress = LSM9DS1_M;
	imu.settings.device.agAddress = LSM9DS1_AG;

	for (int i=4; i<=5; i++) {
		tcaselect(i);
		// The above lines will only take effect AFTER calling
		// imu.begin(), which verifies communication with the IMU
		// and turns it on.
		if (!imu.begin())
		{
			Serial.println("Failed to communicate with LSM9DS1.");
			//Serial.println("Double-check wiring.");
			//Serial.println("Default settings in this sketch will " \
				"work for an out of the box LSM9DS1 " \
				"Breakout, but may need to be modified " \
				"if the board jumpers are.");
			while (1)
				;
		}
	}

	while (!Serial.available())
		;

	current = millis();

	// The loop outside The Loop
	while (1) {
		for (int i=4; i<5; i++) {
			tcaselect(i);

			// read time

			previous = current;
			current  = millis();

			dt = (current - previous) / 1000;

			imu.readGyro();
			imu.readAccel();
			imu.readMag();

			ahrs.updateIMU(imu.ax, imu.ay, imu.az, imu.gx*0.0175, imu.gy*0.0175, imu.gz*0.0175, dt);

			Serial.print(ahrs.getW());
			Serial.print(",");
			Serial.print(ahrs.getX());
			Serial.print(",");
			Serial.print(ahrs.getY());
			Serial.print(",");
			Serial.print(ahrs.getZ());
			Serial.println();

			// delay(PRINT_SPEED);
		}
		Serial.println();
	}
}

void loop()
{}

void printGyro()
{
	// To read from the gyroscope, you must first call the
	// readGyro() function. When this exits, it'll update the
	// gx, gy, and gz variables with the most current data.
	imu.readGyro();

	if (!print_flag) {
		return;
	}

	// Now we can use the gx, gy, and gz variables as we please.
	// Either print them as raw ADC values, or calculated in DPS.
//  Serial.print("G: ");
#ifdef PRINT_CALCULATED
	// If you want to print calculated values, you can use the
	// calcGyro helper function to convert a raw ADC value to
	// DPS. Give the function the value that you want to convert.
	Serial.print(imu.calcGyro(imu.gx), 2);
	Serial.print(", ");
	Serial.print(imu.calcGyro(imu.gy), 2);
	Serial.print(", ");
	Serial.print(imu.calcGyro(imu.gz), 2);
	Serial.print(", ");
//  Serial.println(" deg/s");
#elif defined PRINT_RAW
	Serial.print(imu.gx);
	Serial.print(", ");
	Serial.print(imu.gy);
	Serial.print(", ");
	Serial.print(imu.gz);
	Serial.print(", ");
#endif
}

void printAccel()
{
	// To read from the accelerometer, you must first call the
	// readAccel() function. When this exits, it'll update the
	// ax, ay, and az variables with the most current data.
	imu.readAccel();

	if (!print_flag) {
		return;
	}

	// Now we can use the ax, ay, and az variables as we please.
	// Either print them as raw ADC values, or calculated in g's.
//  Serial.print("A: ");
#ifdef PRINT_CALCULATED
	// If you want to print calculated values, you can use the
	// calcAccel helper function to convert a raw ADC value to
	// g's. Give the function the value that you want to convert.
	Serial.print(imu.calcAccel(imu.ax), 2);
	Serial.print(", ");
	Serial.print(imu.calcAccel(imu.ay), 2);
	Serial.print(", ");
	Serial.print(imu.calcAccel(imu.az), 2);
	Serial.print(", ");
//  Serial.println(" g");
#elif defined PRINT_RAW
	Serial.print(imu.ax);
	Serial.print(", ");
	Serial.print(imu.ay);
	Serial.print(", ");
	Serial.print(imu.az);
	Serial.print(", ");
#endif

}

void printMag()
{
	// To read from the magnetometer, you must first call the
	// readMag() function. When this exits, it'll update the
	// mx, my, and mz variables with the most current data.
	imu.readMag();

	if (!print_flag) {
		return;
	}
	// Now we can use the mx, my, and mz variables as we please.
	// Either print them as raw ADC values, or calculated in Gauss.
//  Serial.print("M: ");
#ifdef PRINT_CALCULATED
	// If you want to print calculated values, you can use the
	// calcMag helper function to convert a raw ADC value to
	// Gauss. Give the function the value that you want to convert.
	Serial.print(imu.calcMag(imu.mx), 2);
	Serial.print(", ");
	Serial.print(imu.calcMag(imu.my), 2);
	Serial.print(", ");
	Serial.print(imu.calcMag(imu.mz), 2);
//  Serial.print(", ")
//  Serial.println(" gauss");
#elif defined PRINT_RAW
	Serial.print(imu.mx);
	Serial.print(", ");
	Serial.print(imu.my);
	Serial.print(", ");
	Serial.print(imu.mz);
//  Serial.print(", ");
#endif
}

// Calculate pitch, roll, and heading.
// Pitch/roll calculations take from this app note:
// http://cache.freescale.com/files/sensors/doc/app_note/AN3461.pdf?fpsp=1
// Heading calculations taken from this app note:
// http://www51.honeywell.com/aero/common/documents/myaerospacecatalog-documents/Defense_Brochures-documents/Magnetic__Literature_Application_notes-documents/AN203_Compass_Heading_Using_Magnetometers.pdf
void printAttitude(float ax, float ay, float az, float mx, float my, float mz)
{
	float roll = atan2(ay, az);
	float pitch = atan2(-ax, sqrt(ay * ay + az * az));

	float heading;
	if (my == 0)
		heading = (mx < 0) ? 180.0 : 0;
	else
		heading = atan2(mx, my);

	heading -= DECLINATION * PI / 180;

	if (heading > PI) heading -= (2 * PI);
	else if (heading < -PI) heading += (2 * PI);
	else if (heading < 0) heading += 2 * PI;

	// Convert everything from radians to degrees:
	heading *= 180.0 / PI;
	pitch *= 180.0 / PI;
	roll  *= 180.0 / PI;

	Serial.print("Pitch, Roll: ");
	Serial.print(pitch, 2);
	Serial.print(", ");
	Serial.println(roll, 2);
	Serial.print("Heading: "); Serial.println(heading, 2);
}
