from .anchor_head import AnchorHead

class GFocalHead(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 ):
        super(GFocalHead, self).__init__()

