# __init__.py

from importlib import resources
from src.model_components import GatedCNN_encoder, TopModel_layer, Decoder_re
from src.wavenet_decoder import Wave_generator, CondNet
from src.model_ensemble import ProtWaveVAE, SS_ProtWaveVAE

# version of the realpython-reader package
__version__ = "0.0.0"



