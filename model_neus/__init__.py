import datetime

from ._utils import log
from .neus import get_model
from .renderer import NeuSRenderer
from .run import run

log(f'\n\n{"="*100}\n\t Model Started at {datetime.datetime.now()}\n')
