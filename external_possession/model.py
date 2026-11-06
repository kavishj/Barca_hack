import torch
from relational_possession.model import RelationalNet


class FrozenNet(RelationalNet):
    def __init__(self,*args,pretrained=None,**kwargs):
        super().__init__(*args,**kwargs)
        if pretrained is not None:
            self.spatial.load_state_dict(torch.load(pretrained,map_location='cpu',weights_only=True))
        for parameter in self.spatial.parameters():parameter.requires_grad_(False)
        self.spatial.eval()

    def train(self,mode=True):
        super().train(mode);self.spatial.eval();return self
