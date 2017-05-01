import functools


class classinstancemethod:
	'''
	A describtor for having a method that is both a class method and an instance method

	Courtesy of: Mike Axiak, from StackOverflow, cheers and thanks for saving us time and effort !
	https://stackoverflow.com/questions/2589690/creating-a-method-that-is-simultaneously-an-instance-and-class-method
	'''
	def __init__(self, method):
		self.method = method

	def __get__(self, obj=None, typ=None):
		@functools.wraps(self.method)
		def _wrapper(*args, **kwargs):
			if obj is not None:
				# as instance method
				return self.method(obj, *args, **kwargs)
			else:
				# as class method
				return self.method(typ, *args, **kwargs)

		return _wrapper

class reorderparams:
	'''
	reorders paramers, specially for instance methods
	can help make static and instance method call identical
	'''
	def __init__ (self, method):
		self.method = method

	def __get__ (self, obj=None, typ=None):
		@functools.wraps(self.method)
		def _wrapper (*args, **kwargs):
			kwargs['typ']  = typ
			kwargs['self'] = obj

			return self.method(*args, **kwargs)

		return _wrapper