from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class VOCDataset(XMLDataset):

    CLASSES = ('mask', 'face', 'helmet', 'person')

    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')
