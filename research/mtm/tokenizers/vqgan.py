import os
import sys

import omegaconf
import torch

import research
from research.mtm.datasets.base import DatasetProtocol
from research.mtm.tokenizers.base import Tokenizer

ROOT_DIR = os.path.dirname(os.path.abspath(research.__file__))
sys.path.append(os.path.join(ROOT_DIR, "../third_party/taming-transformers"))

from ldm.util import instantiate_from_config


class VqganTokenizer(Tokenizer):
    def __init__(
        self,
        rel_config_path: str,
        model_ckpt_path: str,
    ):
        super().__init__()
        model_config_path = os.path.join(
            ROOT_DIR,
            rel_config_path,
        )
        model_config = omegaconf.OmegaConf.load(model_config_path)
        model = instantiate_from_config(model_config.model)

        model.init_from_ckpt(model_ckpt_path)
        model.cuda()
        model.eval()
        self.model = model

    @classmethod
    def create(
        cls,
        key: str,
        train_dataset: DatasetProtocol,
        rel_config_path: str,
        model_ckpt_path: str,
    ) -> "VqganTokenizer":
        return cls(rel_config_path=rel_config_path, model_ckpt_path=model_ckpt_path)

    @property
    def discrete(self) -> bool:
        return True

    def encode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        # check shape is consistant with images
        assert trajectory.dim() == 5
        assert trajectory.min() >= 0
        assert trajectory.max() <= 255

        # normalize trajectory
        trajectory = (trajectory / 255) - 0.5

        B, L, H, W, C = trajectory.shape
        trajectory = trajectory.reshape(B * L, H, W, C)
        trajectory = trajectory.permute(0, 3, 1, 2)
        trajectory = trajectory.to(self.model.dtype).to(self.model.device)

        n_e = self.model.quantize.n_e
        results = []
        with torch.no_grad():
            for i in range(len(trajectory)):
                image = trajectory[i].unsqueeze(0)
                (
                    quant,
                    emb_loss,
                    (perplexity, min_encodings, min_encoding_indices),
                ) = self.model.encode(image)
                one_hot = torch.nn.functional.one_hot(min_encoding_indices, n_e)
                results.append(one_hot.float())
            result = torch.stack(results)
            result = result.reshape(B, L, -1, n_e)
            return result

    def decode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        # check shape is consistant with patches
        assert trajectory.dim() == 4
        # trajectory shape, (B, L, P, C)
        B, L, P, C = trajectory.shape
        # reshape
        trajectory = trajectory.reshape(B * L, P, C)
        side_size = int(P ** (1 / 2))
        embedding_dim = self.model.quantize.embedding.embedding_dim
        shape = [B * L, side_size, side_size, embedding_dim]
        codebook_entries = self.model.quantize.get_codebook_entry(
            trajectory.argmax(dim=-1), shape
        )
        decoded = self.model.decode(codebook_entries)
        decoded = decoded.permute(0, 2, 3, 1)
        decoded = decoded.reshape(B, L, *decoded.shape[1:])
        return decoded
