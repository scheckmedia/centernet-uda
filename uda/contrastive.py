from uda.base import Model
import torch
from torch import nn
import torch.nn.functional as F
from utils.helper import CustomDataParallel, load_model, save_model
from pathlib import Path


class ContrastiveSimSiam(Model):
    def __init__(self, source_weight=0.5,
                 target_weight=0.5, warmup_steps=5000):
        super().__init__()
        self.weights = [source_weight, target_weight]
        self.warmup_steps = warmup_steps
        self.step_cnt = 0

    def init_done(self):
        super().init_done()
        self.projection = Projection(
            self.backend.dim_feature_extractor,
            self.backend.dim_feature_extractor,
            self.backend.dim_feature_extractor)
        self.prediction = Prediction(
            self.backend.dim_feature_extractor,
            self.backend.dim_feature_extractor,
            self.backend.dim_feature_extractor)

    def to(self, device, parallel=False):
        super().to(device, parallel)

        if parallel:
            self.projection = CustomDataParallel(self.projection)
            self.prediction = CustomDataParallel(self.prediction)

        self.projection.to(device)
        self.prediction.to(device)

    def step(self, data, is_training=True):
        for k in data:
            if not isinstance(data[k], torch.Tensor):
                continue

            data[k] = data[k].to(device=self.device, non_blocking=True)

        source = data["input"]
        target = data["target_domain_input"]

        stats = {}
        outputs = {}

        if is_training:
            self.optimizer.zero_grad()

            if self.step_cnt > self.warmup_steps:
                outputs['source_domain'] = self.backend(source, head_only=True)
                c_loss, c_stats = self.centernet_loss(
                    outputs['source_domain'], data)
                c_loss.backward()
            else:
                c_stats = {}
                c_loss = 0.0

            source_contrastive = data["input_contrastive"]
            target_contrastive = data["target_domain_input_contrastive"]

            if torch.rand(1) > 0.5:
                x1 = source
                x2 = source_contrastive
                w = self.weights[0]
            else:
                x1 = target
                x2 = target_contrastive
                w = self.weights[1]

            features_x1 = self.backend.base(x1)
            features_x2 = self.backend.base(x2)

            f1 = torch.flatten(F.adaptive_avg_pool2d(features_x1, 1), 1)
            f2 = torch.flatten(F.adaptive_avg_pool2d(features_x2, 1), 1)
            z1, z2 = self.projection(f1), self.projection(f2)
            p1, p2 = self.prediction(z1), self.prediction(z2)

            d1 = -F.cosine_similarity(p1, z2.detach(), dim=-1).mean() * 0.5
            d2 = -F.cosine_similarity(p2, z1.detach(), dim=-1).mean() * 0.5

            loss = (d1 + d2) * w
            loss.backward()
            stats['cosine_similarity'] = loss

            self.optimizer.step()
            self.step_cnt += 1
        else:
            outputs['source_domain'] = self.backend(source)
            c_loss, c_stats = self.centernet_loss(
                outputs['source_domain'], data)

        if not is_training:
            loss = 0.0  # for non-training we have no contrastive loss

        stats = {**c_stats, **stats}
        stats["total_loss"] = c_loss + loss

        for s in stats:
            stats[s] = stats[s].cpu().detach()

        outputs["stats"] = stats

        return outputs

    def save_model(self, path, epoch, with_optimizer=False):
        super().save_model(path, epoch, with_optimizer)
        alias = 'last' if 'last' in path else 'best'

        save_model(self.projection, str(
            Path(path).with_name(f'{alias}_projection.pth')), epoch)
        save_model(self.prediction, str(
            Path(path).with_name(f'{alias}_prediction.pth')), epoch)

    def load_model(self, path, resume=False):
        alias = 'last' if 'last' in path else 'best'
        load_model(
            self.projection,
            None,
            None,
            str(Path(path).with_name(f'{alias}_prediction.pth')),
            resume)
        load_model(
            self.prediction,
            None,
            None,
            str(Path(path).with_name(f'{alias}_prediction.pth')),
            resume)
        return super().load_model(path, resume=resume)

# borrowed from
# https://github.com/PatrickHua/SimSiam/blob/main/models/simsiam.py


class Projection(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out-
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d.
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class Prediction(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers.
        The dimension of h’s input and output (z and p) is d = 2048,
        and h’s hidden layer’s dimension is 512, making h a
        bottleneck structure (ablation in supplement).
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing.
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
