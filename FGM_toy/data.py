from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ToyAnchors:
    audio: np.ndarray
    visual: np.ndarray


class ToyDataset(Dataset):
    def __init__(self, xA: np.ndarray, xB: np.ndarray, y: np.ndarray, zA: np.ndarray, zB: np.ndarray) -> None:
        self.xA = torch.from_numpy(xA.astype(np.float32))
        self.xB = torch.from_numpy(xB.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.zA = torch.from_numpy(zA.astype(np.int64))
        self.zB = torch.from_numpy(zB.astype(np.int64))

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "audio": self.xA[index],
            "visual": self.xB[index],
            "label": self.y[index],
            "zA": self.zA[index],
            "zB": self.zB[index],
        }


def sample_latent(n: int, s: float, eta: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode = (rng.random(n) < s).astype(np.int64)
    a_bit = rng.integers(0, 2, size=n)
    b_bit = np.where(mode == 1, rng.integers(0, 2, size=n), a_bit)
    base = np.where(mode == 1, a_bit ^ b_bit, a_bit)
    flip = (rng.random(n) < eta) & (mode == 1)
    y = np.where(flip, 1 - base, base)
    return (mode * 2 + a_bit).astype(np.int64), (mode * 2 + b_bit).astype(np.int64), y.astype(np.int64)


def make_anchors(d: int = 16, seed: int = 0) -> ToyAnchors:
    rng = np.random.default_rng(seed)
    return ToyAnchors(audio=rng.standard_normal((4, d)), visual=rng.standard_normal((4, d)))


def embed(z: np.ndarray, anchors: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return anchors[z] + sigma * rng.standard_normal((len(z), anchors.shape[1]))


def make_dataset(
    n: int,
    s: float,
    eta: float = 0.0,
    d: int = 16,
    sigma_A: float = 0.3,
    sigma_B: float = 0.8,
    seed: int = 0,
    anchors: ToyAnchors | None = None,
) -> ToyDataset:
    rng = np.random.default_rng(seed)
    anchors = anchors or make_anchors(d=d, seed=seed + 10_000)
    zA, zB, y = sample_latent(n, s, eta, rng)
    xA = embed(zA, anchors.audio, sigma_A, rng)
    xB = embed(zB, anchors.visual, sigma_B, rng)
    return ToyDataset(xA, xB, y, zA, zB)


def create_loaders(
    n_train: int,
    n_val: int,
    s: float,
    eta: float,
    batch_size: int,
    seed: int,
    d: int = 16,
    sigma_A: float = 0.3,
    sigma_B: float = 0.8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    anchors = make_anchors(d=d, seed=seed + 10_000)
    train = make_dataset(n_train, s, eta, d, sigma_A, sigma_B, seed, anchors)
    val = make_dataset(n_val, s, eta, d, sigma_A, sigma_B, seed + 1, anchors)
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def joint_table(s: float, eta: float = 0.0) -> np.ndarray:
    P = np.zeros((4, 4, 2), dtype=np.float64)
    P[0, 0, 0] += (1 - s) / 2
    P[1, 1, 1] += (1 - s) / 2
    for za, zb, base_y in ((2, 2, 0), (2, 3, 1), (3, 2, 1), (3, 3, 0)):
        P[za, zb, base_y] += s / 4 * (1 - eta)
        P[za, zb, 1 - base_y] += s / 4 * eta
    if abs(P.sum() - 1.0) >= 1e-9:
        raise ValueError("joint table does not sum to 1")
    return P


def _cmi_from_joint(P: np.ndarray, eps: float = 1e-12) -> float:
    pYB = P.sum(axis=0)
    pY_B = pYB / (pYB.sum(axis=1, keepdims=True) + eps)
    H_Y_B = -(pYB * np.log2(pY_B + eps)).sum()
    pAB = P.sum(axis=2)
    pY_AB = P / (pAB[:, :, None] + eps)
    H_Y_AB = -(P * np.log2(pY_AB + eps)).sum()
    return float(H_Y_B - H_Y_AB)


def cmi_A_given_B(s: float, eta: float = 0.0) -> float:
    return _cmi_from_joint(joint_table(s, eta))


def cmi_B_given_A(s: float, eta: float = 0.0) -> float:
    return _cmi_from_joint(joint_table(s, eta).transpose(1, 0, 2))


def Hb(eta: float) -> float:
    if eta <= 0.0 or eta >= 1.0:
        return 0.0
    return float(-eta * np.log2(eta) - (1 - eta) * np.log2(1 - eta))


def theoretical_single_acc(s: float) -> float:
    return 1.0 - s / 2.0


def theoretical_joint_acc(s: float, eta: float) -> float:
    return 1.0 - s * eta


if __name__ == "__main__":
    for s_value in (0.0, 0.5, 1.0):
        for eta_value in (0.0, 0.1, 0.3):
            want = s_value * (1 - Hb(eta_value))
            assert abs(cmi_A_given_B(s_value, eta_value) - want) < 1e-9
            assert abs(cmi_B_given_A(s_value, eta_value) - want) < 1e-9
    print("CMI enumeration matches the closed form.")

