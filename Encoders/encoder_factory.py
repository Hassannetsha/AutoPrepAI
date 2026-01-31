from .onehot_encoder import OneHotEncoderStrategy
from .label_encoder import LabelEncoderStrategy
from .target_encoder import TargetEncoderStrategy

class EncoderFactory:

    _encoders = {
        "onehot": OneHotEncoderStrategy(),
        "one-hot": OneHotEncoderStrategy(),
        "label": LabelEncoderStrategy(),
        "target": TargetEncoderStrategy(),
    }

    @staticmethod
    def get_encoder(method: str):
        method = method.lower()
        for key in EncoderFactory._encoders:
            if method.startswith(key):
                return EncoderFactory._encoders[key]
        return OneHotEncoderStrategy()  # default
