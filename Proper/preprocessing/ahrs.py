from math import asin, atan2, atan2, pi, sqrt

class AHRS:
	"""AHRS.cpp Python edition"""
	def __init__(self, q0=1, q1=0, q2=0, q3=0):
		self.q0 = q0
		self.q1 = q1
		self.q2 = q2
		self.q3 = q3

		self.twoKi       = 0.0
		self.twoKp       = 0.0
		self.gyroOffset  = [0.0, 0.0, 0.0]
		self.integralFBx = 0.0
		self.integralFBy = 0.0
		self.integralFBz = 0.0

	def update (self, ax, ay, az, gx, gy, gz, mx, my, mz, dt):
		recipNorm
		q0q0 = q0q1 = q0q2 = q0q3 = q1q1 = q1q2 = q1q3 = q2q2 = q2q3 = q3q3 = \
		hx = hy = bx = bz = halfvx = halfvy = halfvz = halfwx = halfwy = halfwz = \
		halfex = halfey = halfez = qa = qb = qc = 0.0

		# Use IMU algorithm if magnetometer measurement invalid (avoids NaN in magnetometer normalisation)
		if((mx == 0.0) and (my == 0.0) and (mz == 0.0)):
			updateIMU(gx, gy, gz, ax, ay, az, dt)
			return

		# Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
		if(not ((ax == 0.0) and (ay == 0.0) and (az == 0.0))):
			# Normalise accelerometer measurement
			recipNorm = self.invSqrt(ax * ax + ay * ay + az * az)
			ax *= recipNorm
			ay *= recipNorm
			az *= recipNorm

			# Normalise magnetometer measurement
			recipNorm = self.invSqrt(mx * mx + my * my + mz * mz)
			mx *= recipNorm
			my *= recipNorm
			mz *= recipNorm

			# Auxiliary variables to avoid repeated arithmetic
			q0q0 = self.q0 * self.q0
			q0q1 = self.q0 * self.q1
			q0q2 = self.q0 * self.q2
			q0q3 = self.q0 * self.q3
			q1q1 = self.q1 * self.q1
			q1q2 = self.q1 * self.q2
			q1q3 = self.q1 * self.q3
			q2q2 = self.q2 * self.q2
			q2q3 = self.q2 * self.q3
			q3q3 = self.q3 * self.q3

			# Reference direction of Earth's magnetic field
			hx = 2.0 * (mx * (0.5 - q2q2 - q3q3) + my * (q1q2 - q0q3) + mz * (q1q3 + q0q2))
			hy = 2.0 * (mx * (q1q2 + q0q3) + my * (0.5 - q1q1 - q3q3) + mz * (q2q3 - q0q1))
			bx = sqrt(hx * hx + hy * hy)
			bz = 2.0 * (mx * (q1q3 - q0q2) + my * (q2q3 + q0q1) + mz * (0.5 - q1q1 - q2q2))

			# Estimated direction of gravity and magnetic field
			halfvx = q1q3 - q0q2
			halfvy = q0q1 + q2q3
			halfvz = q0q0 - 0.5 + q3q3
			halfwx = bx * (0.5 - q2q2 - q3q3) + bz * (q1q3 - q0q2)
			halfwy = bx * (q1q2 - q0q3) + bz * (q0q1 + q2q3)
			halfwz = bx * (q0q2 + q1q3) + bz * (0.5 - q1q1 - q2q2)

			# Error is sum of cross product between estimated direction and measured direction of field vectors
			halfex = (ay * halfvz - az * halfvy) + (my * halfwz - mz * halfwy)
			halfey = (az * halfvx - ax * halfvz) + (mz * halfwx - mx * halfwz)
			halfez = (ax * halfvy - ay * halfvx) + (mx * halfwy - my * halfwx)

			# Compute and apply integral feedback if enabled
			if(self.twoKi > 0.0):
				self.integralFBx += self.twoKi * halfex * dt	# integral error scaled by Ki
				self.integralFBy += self.twoKi * halfey * dt
				self.integralFBz += self.twoKi * halfez * dt
				gx += self.integralFBx	# apply integral feedback
				gy += self.integralFBy
				gz += self.integralFBz
			else:
				self.integralFBx = 0.0	# prevent integral windup
				self.integralFBy = 0.0
				self.integralFBz = 0.0

			# Apply proportional feedback
			gx += self.twoKp * halfex
			gy += self.twoKp * halfey
			gz += self.twoKp * halfez

		# Integrate rate of change of quaternion
		gx *= (0.5 * dt)		# pre-multiply common factors
		gy *= (0.5 * dt)
		gz *= (0.5 * dt)
		qa = q0
		qb = q1
		qc = q2
		self.q0 += (-qb * gx - qc * gy - self.q3 * gz)
		self.q1 += (qa * gx + qc * gz - self.q3 * gy)
		self.q2 += (qa * gy - qb * gz + self.q3 * gx)
		self.q3 += (qa * gz + qb * gy - qc * gx)

		# Normalise quaternion
		recipNorm = self.invSqrt(self.q0 * self.q0 + self.q1 * self.q1 + self.q2 * self.q2 + self.q3 * self.q3)
		self.q0 *= recipNorm
		self.q1 *= recipNorm
		self.q2 *= recipNorm
		self.q3 *= recipNorm

	def updateIMU (self, ax, ay, az, gx, gy, gz, dt):
		recipNorm = \
		halfvx = halfvy = halfvz = \
		halfex = halfey = halfez = \
		qa = qb = qc = 0.0

		gx -= self.gyroOffset[0]
		gy -= self.gyroOffset[1]
		gz -= self.gyroOffset[2]

		# Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
		if(not ((ax == 0.0) and (ay == 0.0) and (az == 0.0))):
			# Normalise accelerometer measurement
			recipNorm = self.invSqrt(ax * ax + ay * ay + az * az)
			ax *= recipNorm
			ay *= recipNorm
			az *= recipNorm

			# Estimated direction of gravity and vector perpendicular to magnetic flux
			halfvx = self.q1 * self.q3 - self.q0 * self.q2
			halfvy = self.q0 * self.q1 + self.q2 * self.q3
			halfvz = self.q0 * self.q0 - 0.5 + self.q3 * self.q3

			# Error is sum of cross product between estimated and measured direction of gravity
			halfex = (ay * halfvz - az * halfvy)
			halfey = (az * halfvx - ax * halfvz)
			halfez = (ax * halfvy - ay * halfvx)

			# Compute and apply integral feedback if enabled
			if(self.twoKi > 0.0):
				self.integralFBx += self.twoKi * halfex * dt	# integral error scaled by Ki
				self.integralFBy += self.twoKi * halfey * dt
				self.integralFBz += self.twoKi * halfez * dt
				gx += self.integralFBx	# apply integral feedback
				gy += self.integralFBy
				gz += self.integralFBz
			else:
				self.integralFBx = 0.0	# prevent integral windup
				self.integralFBy = 0.0
				self.integralFBz = 0.0

			# Apply proportional feedback
			gx += self.twoKp * halfex
			gy += self.twoKp * halfey
			gz += self.twoKp * halfez

		# Integrate rate of change of quaternion
		gx *= (0.5 * dt)		# pre-multiply common factors
		gy *= (0.5 * dt)
		gz *= (0.5 * dt)
		qa = self.q0
		qb = self.q1
		qc = self.q2
		self.q0 += (-qb * gx - qc * gy - self.q3 * gz)
		self.q1 += (qa * gx + qc * gz - self.q3 * gy)
		self.q2 += (qa * gy - qb * gz + self.q3 * gx)
		self.q3 += (qa * gz + qb * gy - qc * gx)

		# Normalise quaternion
		recipNorm = self.invSqrt(self.q0 * self.q0 + self.q1 * self.q1 + self.q2 * self.q2 + self.q3 * self.q3)
		self.q0 *= recipNorm
		self.q1 *= recipNorm
		self.q2 *= recipNorm
		self.q3 *= recipNorm

		# returns (W,X,Y,Z)
		return (self.q0, self.q1, self.q2, self.q3)

	def setGyroOffset (self, offsetX, offsetY, offsetZ):
		self.gyroOffset[0] = offsetX
		self.gyroOffset[1] = offsetY
		self.gyroOffset[2] = offsetZ

	def getEuler (self):
		yaw   = atan2(
				2*(self.q0*self.q3+self.q1*self.q2),
				1 - 2*(self.q2*self.q2+self.q3*self.q3)
			) * (180.0/pi)

		roll  = atan2(
				2*(self.q0*self.q1+self.q2*self.q3),
				1 - (2*self.q1*self.q1+self.q2*self.q2)
			) * (180.0/pi)

		pitch = asin(2*(self.q0*self.q2-self.q3*self.q1)) * (180.0/pi)

		return (roll, pitch, yaw)

	def invSqrt (self, x):
		return 1 / sqrt(x)

	def getW (self):
		return self.q0

	def getX (self):
		return self.q1

	def getY (self):
		return self.q2

	def getZ (self):
		return self.q3

if __name__ == '__main__':
	print('hello')
