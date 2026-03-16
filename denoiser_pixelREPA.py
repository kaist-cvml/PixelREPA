import torch
import torch.nn as nn
from model_pixelREPA import JiT_models, MaskedTransformerAdapter_models
import torch.nn.functional as F
from timm import create_model


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )
    

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class Denormalize(nn.Module):
    def __init__(self, mean, std):
        super(Denormalize, self).__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return x * self.std + self.mean


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        
        self.repa_model = create_model(args.repa_model, pretrained=True, img_size=args.repa_image_size, patch_size=args.repa_patch_size)
        for param in self.repa_model.parameters():
            param.requires_grad = False
        self.repa_model.eval()
        repa_z_dim = self.repa_model.embed_dim
        self.repa_z_dim = repa_z_dim
        self.repa_image_size = args.repa_image_size
        self.de_scale = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.masked_transformer_adapter = MaskedTransformerAdapter_models[args.mta](
            input_size=args.img_size,
            in_channels=3,
            out_channels=repa_z_dim,
            num_classes=args.class_num,
        )
        
        self.patch_size = self.net.patch_size
        self.patch_mask_ratio = args.patch_mask_ratio
        
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def adjust(self, x):
        return torch.nn.functional.interpolate(x, 224, mode='bicubic')

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)
        x_encoded = self.net.forward_encoder(z, t.flatten(), labels_dropped)
        
        return self.forward_jit(x_encoded, z, v, labels_dropped, t), self.forward_mta(x, x_encoded, labels_dropped, t)

    def forward_jit(self, x_encoded, z, v, labels_dropped, t):
        x_pred = self.net.forward_decoder(x_encoded, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        
        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()
        
        return loss
    
    def forward_mta(self, x, x_encoded, labels_dropped, t):
        N, C, H, W = x.shape
        P = self.patch_size

        # prepare repa feature
        rescale_x = self.scale(self.de_scale(x))
        if (H, W) != (self.repa_image_size, self.repa_image_size):
            repa_x = F.interpolate(
                rescale_x,
                size=(self.repa_image_size, self.repa_image_size),
                mode="bilinear",
                align_corners=False
            )
        else:
            repa_x = rescale_x
        
        z_dino = self.repa_model.forward_features(repa_x)[:, self.repa_model.num_prefix_tokens:]
        
        mask_ratio = torch.full((N, 1, 1, 1), self.patch_mask_ratio, device=x.device)

        h_grid, w_grid = H // P, W // P
        noise_grid = torch.rand(N, 1, h_grid, w_grid, device=x.device)
        
        mask_patch = (noise_grid > mask_ratio).float()
        mask_patch = mask_patch.reshape(N, h_grid*w_grid, 1)

        x_encoded_masked = torch.where(mask_patch.bool(), x_encoded, self.net.mask_token)
        x_decoded_mta = self.masked_transformer_adapter(x_encoded_masked, t.flatten(), labels_dropped)
        
        z_dino = F.normalize(z_dino, dim=-1)
        x_decoded_mta = F.normalize(x_decoded_mta, dim=-1)
        cos_sim = F.cosine_similarity(
            x_decoded_mta,
            z_dino,
            dim=-1
        )
        
        diff_mta = 1.0 - cos_sim 
        loss_mta = diff_mta.mean()
        
        return loss_mta

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)[0]
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)[0]
        return z
    
    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next, v_pred

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next, v_pred
    
    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.net.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
