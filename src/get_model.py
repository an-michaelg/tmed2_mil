# compcars model architecture, featuring resnet encoding layer and attention

import torch
import torch.nn as nn
from models.wrn import build_WideResNet
DIM_EMBEDDING = 128
DIM_HIDDEN = 256
RESOLUTION = 224

def count_module_params(module, title):
    num_parameters = sum([x.nelement() for x in module.parameters()])
    print(f"The number of parameters in the {title}: {num_parameters/1000:9.2f}k")
    
# returns a feature extractor that extends nn.Module
def get_encoder(arch, resolution, embedding_dim, pre=True):
    if arch == 'Densenet121':
        assert resolution == 224
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=pre)
        num_ftrs = model.classifier.in_features
        # change the final linear layer for transfer learning
        model.classifier = nn.Linear(num_ftrs, embedding_dim)
    elif arch == 'WideResnet-28-2':
        wrn_builder = build_WideResNet(first_stride=1, 
                                       depth=28,
                                       widen_factor=2,
                                       drop_rate=0.5)
        model = wrn_builder.build(num_classes=embedding_dim)
    count_module_params(model, 'encoder')
    return model

class GatedAttention(nn.Module): # GA module (https://arxiv.org/pdf/1802.04712.pdf)
    def __init__(self, input_dim, hidden_dim, use_gated=False, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.V = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)
        if use_gated:
            self.U = nn.Linear(input_dim, hidden_dim)
        else:
            self.U = None
            
    def forward(self, x): # assume x is Dx1, where D = input_dim 
        hidden = torch.tanh(self.V(x))
        if self.U:
            gate = torch.tanh(self.U(x))
            hidden = hidden * gate # hadamard product
        a = self.w(hidden) # dot product 
        return a
    
class DenseMLP(nn.Module): # simple dense MLP for classification
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        hidden = torch.relu(self.W1(x))
        logits = self.W2(hidden)
        return logits

class GatedAttentionClassifier(nn.Module):
    def __init__(self, arch, img_W, embedding_dim, dense_hidden_dim, num_classes, attn, 
                 encoder_pretrained=True, use_gated=False, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.encoder = get_encoder(arch, img_W, embedding_dim, encoder_pretrained)
        self.dense_mlp = DenseMLP(embedding_dim, dense_hidden_dim, num_classes)
        self.use_attn = attn
        if self.use_attn:
            self.attention = GatedAttention(embedding_dim, dense_hidden_dim, use_gated)
    
    def forward(self, x_list):
        # expected to take in an arbitrary length N list of Bx3xHxW
        logits, attns = [], []
        for i, x in enumerate(x_list):
            emb = self.encoder(x) # BxD
            logits.append(self.dense_mlp(emb)) # BxC 
            if self.use_attn:
                attns.append(self.attention(emb)) # Bx1
        if self.use_attn:
            # normalizing attention
            attns = torch.stack(attns) # NxBx1
            # norm = torch.softmax(attns, dim=0) # NxBx1 (normalized across N dimension)
        else:
            attns = None
        # # aggregating logits based on attention weighting
        logits = torch.stack(logits)  # NxBxC
        # weighted_logits = norm * logits # NxBxC
        # agg_logits = torch.sum(weighted_logits, dim=0) # BxC
        return logits, attns
    
def gac_unit_test(use_cuda=True):
    resolution = 112
    gac = GatedAttentionClassifier(arch='WideResnet-28-2',
                                   img_W=resolution,
                                   embedding_dim=DIM_EMBEDDING,
                                   dense_hidden_dim=DIM_HIDDEN,
                                   num_classes=5,
                                   attn=False,
                                   encoder_pretrained=True,
                                   use_gated=True)
    
    num_parameters = sum([x.nelement() for x in gac.parameters()])
    print(f"The number of parameters in the GAC: {num_parameters/1000:9.2f}k")
    
    batch_size = 2
    list_length = 64
    test_in = [torch.randn((batch_size, 3, resolution, resolution)) for _ in range(list_length)]
    if use_cuda:
        if torch.cuda.is_available():
            print('Using GPU acceleration: GPU{}'.format(torch.cuda.current_device()))
            gac = gac.cuda()
            test_in = [k.cuda() for k in test_in]
        else:
            print('CUDA is not available. Using CPU')
    logits, attns = gac(test_in)
    return logits, attns # should return a BxC tensor on CUDA device

if __name__ == '__main__':
    l = gac_unit_test()
        
        
        
        
        
        