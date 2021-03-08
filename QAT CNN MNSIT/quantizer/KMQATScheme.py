from enum import Enum, unique

@unique
class KMQATScheme(Enum):
    FullPrecision = 0
    Dequantization = 1
    ErrorCompensation = 2
    LossAwareCompensation = 3
    Other = 4
