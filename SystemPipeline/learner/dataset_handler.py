import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


from features.features_transformer import FeaturesTransformer
from utils.decorators import classinstancemethod

class DatasetHandler:
	_FEATURE_VECTOR_MAKER = FeaturesTransformer