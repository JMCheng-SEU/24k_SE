# This source code is licensed under the license found in the
# LICENSE-MIT.txt file in the root directory of this source tree.
import fairseq
import torch
from omegaconf import ListConfig
from model.ASR_loss.features_config import FeaturesConfig

class huBERT:
    def __init__(self, model_path, layer, device='cuda'):
        super().__init__()
        self.path = model_path
        self.layer = [la for la in layer] if isinstance(layer, ListConfig) else layer

        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.path])
        self.model = models[0]
        self.model = self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_feats(self, x):
        if isinstance(self.layer, list):
            feats = torch.stack([self.model.extract_features(source=x.squeeze(1), padding_mask=None, mask=False,
                                                 output_layer=layer)[0].detach() for layer in self.layer], dim=0)
        else:
            feats, _ = self.model.extract_features(source=x.squeeze(1), padding_mask=None, mask=False,
                                                   output_layer=self.layer)
            feats = feats.detach()
        return feats


def load_lexical_model(model_name, lexical_path, device="cuda", sr=16000, layer=6):
    if model_name.lower() == 'hubert':
        ret = huBERT(lexical_path, layer)
        ret.model.to(device)
        return ret
#
# features_config = FeaturesConfig()
# ft_model = load_lexical_model(features_config.feature_model,
#                                    features_config.state_dict_path,
#                                    device="cuda", sr=16000)
# inputs = torch.randn(8, 16000).to('cuda')
# cond_features = ft_model.extract_feats(inputs)
#
#
# print(cond_features.shape)
