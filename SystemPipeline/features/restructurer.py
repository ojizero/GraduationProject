import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


from utils.decorators import classinstancemethod
from features.features_transformer import FeaturesTransformer

class StructureTransformer (FeaturesTransformer):
	@classinstancemethod
	def transform (obj, **kwargs):
		names, values = super().transform(**kwargs)
		# do transfomation here
