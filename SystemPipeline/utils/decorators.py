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

# preserve original staticmethod
staticmethod_ = staticmethod

# extend staticmethod, implementing the `__call__` function
class staticmethod (staticmethod_):
	'''
		Custom staticmethod class implementing `__call__` method
		used to unify calling staticmethods and instancemethods
		within the Extractor class using simply the `__call__` function
	'''
	def __call__ (self_, *args, **kwargs):
		return self_.__func__(*args, **kwargs)


class instancemethod:
	'''
		Implementing __func__ method on instance methods
	'''
	def __init__ (self, method):
		self.method = method

	def __get__ (self, obj=None, typ=None):
		@functools.wraps(self.method)
		def _method (*args, **kwargs):
			return self.method(obj, *args, **kwargs)

		return _method

	def __func__ (self_, *args, **kwargs):
		self_.method(*args, **kwargs)

