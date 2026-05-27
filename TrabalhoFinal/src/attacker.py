"""Rede atacante: gera uma perturbacao de K pixels em imagens RGB.

Une os dois artigos de referencia:

- Su et al. (2019), "One Pixel Attack..." propoe modificar K pixels (1, 3 ou 5)
  para enganar a rede. No artigo eles usam Evolucao Diferencial (caixa preta);
  aqui substituimos o otimizador evolutivo por uma rede neural treinada via
  back-propagation atraves do classificador (white-box adversarial training,
  no espirito do FGSM de Huang et al. 2017).

- Huang et al. (2017) usa FGSM (gradiente do classificador em relacao a
  entrada) para gerar perturbacoes em todos os pixels. Aqui mantemos esse
  acesso: o gradiente da perda do classificador em relacao a entrada e
  ALIMENTADO como canal adicional do atacante, dando-lhe a informacao de
  saliencia (similar ao JSMA), e restringimos a perturbacao a K pixels via
  Gumbel-Softmax sem reposicao (top-K Plackett-Luce) para tornar a selecao
  discreta diferenciavel.

Em imagens RGB, "1 pixel" segue a convencao do paper: uma posicao (x, y) com
os 3 canais (R, G, B) substituidos simultaneamente, totalizando 5 graus de
liberdade por pixel atacado.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class KPixelAttacker(nn.Module):
    """Gera uma perturbacao de K pixels com base em (x, gradiente).

    Arquitetura totalmente convolucional para preservar a estrutura espacial,
    evitando o "mode collapse" para uma posicao fixa que ocorre quando o
    decoder e Linear sobre features achatadas.

    Entradas:
      x:    imagem original (B, C, H, W) em [0, 1].
      grad: gradiente da perda do classificador em relacao a x (mesmo shape).

    Saidas:
      x_adv:     imagem perturbada = x * (1 - mask) + value_map * mask.
      mask:      tensor (B, 1, H, W) aproximadamente K-hot (broadcast para C canais).
      value_map: tensor (B, C, H, W) com o valor candidato em cada posicao.
    """

    def __init__(
        self,
        image_size: int = 32,
        num_channels: int = 3,
        num_pixels: int = 1,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_pixels = num_pixels

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.position_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_head = nn.Conv2d(hidden_channels, num_channels, kernel_size=1)

    def _sample_topk_mask(
        self, logits: torch.Tensor, tau: float, hard: bool
    ) -> torch.Tensor:
        """Amostragem Gumbel-Softmax sem reposicao (Plackett-Luce relaxado)."""
        running_logits = logits
        mask_sum = torch.zeros_like(logits)
        for _ in range(self.num_pixels):
            m = F.gumbel_softmax(running_logits, tau=tau, hard=hard, dim=-1)
            mask_sum = mask_sum + m
            running_logits = running_logits - m.detach() * 1e6
        return mask_sum.clamp(max=1.0)

    def _produce_logits_and_values(
        self, x: torch.Tensor, grad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([x, grad], dim=1)
        feats = self.encoder(inp)
        pos_logits = self.position_head(feats).flatten(1)
        value_map = torch.sigmoid(self.value_head(feats))
        return pos_logits, value_map

    def forward(
        self,
        x: torch.Tensor,
        grad: torch.Tensor,
        tau: float = 0.5,
        hard: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_logits, value_map = self._produce_logits_and_values(x, grad)
        mask_flat = self._sample_topk_mask(pos_logits, tau=tau, hard=hard)
        mask = mask_flat.view(-1, 1, self.image_size, self.image_size)
        x_adv = x * (1.0 - mask) + value_map * mask
        return x_adv, mask, value_map

    @torch.no_grad()
    def attack_deterministic(
        self, x: torch.Tensor, grad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Geracao deterministica usada na avaliacao (top-K em vez de Gumbel)."""
        pos_logits, value_map = self._produce_logits_and_values(x, grad)
        topk_idx = pos_logits.topk(self.num_pixels, dim=-1).indices
        mask_flat = torch.zeros_like(pos_logits)
        mask_flat.scatter_(1, topk_idx, 1.0)
        mask = mask_flat.view(-1, 1, self.image_size, self.image_size)
        x_adv = x * (1.0 - mask) + value_map * mask
        return x_adv, mask, value_map


def compute_input_gradient(
    classifier: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Calcula d(CE)/dx (gradiente da perda em relacao a entrada).

    Usado como canal adicional do atacante (saliencia ao estilo FGSM/JSMA).
    Normalizamos por imagem para estabilizar o treinamento.
    """
    x_grad = x.detach().clone().requires_grad_(True)
    logits = classifier(x_grad)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_grad)[0].detach()
    flat = grad.view(grad.size(0), -1)
    max_abs = flat.abs().amax(dim=1).clamp(min=1e-8)
    grad = grad / max_abs.view(-1, 1, 1, 1)
    return grad


def margin_loss(logits: torch.Tensor, y: torch.Tensor, kappa: float = 5.0) -> torch.Tensor:
    """Perda Carlini-Wagner para ataque nao-direcionado.

    Queremos que a classe verdadeira deixe de ser a maxima. Quando o ataque
    ja teve sucesso a perda e travada em -kappa para nao seguir empurrando
    indefinidamente, focando a otimizacao em exemplos ainda nao quebrados.
    """
    true_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)
    other = logits.clone()
    other.scatter_(1, y.unsqueeze(1), float("-inf"))
    max_other = other.max(dim=1).values
    return torch.clamp(true_logit - max_other, min=-kappa).mean()
