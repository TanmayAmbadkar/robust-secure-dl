import torch
import torch.nn as nn
import torch.nn.functional as F
from CNN.resnet import ResNet18

class IntermediateRep(nn.Module):
    
    def __init__(self, resnet, layer_to_extract):
        super(IntermediateRep, self).__init__()
        self.resnet = resnet
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.layer_to_extract = layer_to_extract
        
    def forward(self, x):
        
        resnet_out = torch.argmax(F.softmax(self.resnet(x), dim=1))
        for name, layer in self.resnet.named_children():
            x = layer(x)
            if name == self.layer_to_extract:
                break
        
        return x, resnet_out

def create_model(layer = "layer1"):
    
    model = ResNet18()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load('./model_weights/cpu_model.pth'))
    model.to(device)
    model.eval()
    
    intermediate_rep = IntermediateRep(model, layer).cuda().eval()
    intermediate_rep.eval()
    
    return intermediate_rep