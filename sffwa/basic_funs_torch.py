import torch


class Sphere:
    def __init__(self, n_dim: int, device: torch.device):
        self.device = device
        self.n_dim = n_dim  # [self.n_dim]

    def __call__(self, X: torch.Tensor):
        Y = torch.sum(X**2, dim=-1)
        return Y


class BaseEllipsoid:
    def __init__(self, n_dim: int, device: torch.device):
        self.device = device
        self.n_dim = n_dim
        self.coefs = torch.pow(
            1e6, torch.arange(
                self.n_dim, device=self.device) / (self.n_dim - 1)
        )  # [self.n_dim]

    def __call__(self, X: torch.Tensor):
        Y = self.coefs * (X**2)
        return Y


class BaseBentCigar:
    def __init__(self, n_dim: int, device: torch.device):
        self.n_dim = n_dim
        self.device = device

    def __call__(self, X: torch.Tensor):
        Y = X[:, 0] ** 2 + 1e6 * torch.sum(X[:, 1:] ** 2, dim=-1)
        return Y.view((-1, 1))


class Ellipsoid:
    def __init__(self, n_dim: int, device: torch.device):
        self.f = BaseEllipsoid(n_dim, device)

    def __call__(self, X: torch.Tensor):
        return self.f(X).sum(dim=-1)


class BentCigar:
    def __init__(self, n_dim: int, device: torch.device):
        self.f = BaseBentCigar(n_dim, device)

    def __call__(self, X: torch.Tensor):
        return self.f(X).sum(dim=-1)


class Discus:
    def __init__(self, n_dim: int, device=torch.device):
        self.n_dim = n_dim
        self.device = device

    def __call__(self, X):
        Y = 1e6 * X[:, 0] ** 2 + torch.sum(X[:, 1:] ** 2, dim=-1)
        return Y


class DiffPowers:
    def __init__(self, n_dim: int, device=torch.device):
        self.n_dim = n_dim
        self.device = device

    def __call__(self, X):
        Y = torch.sum(
            torch.pow(
                torch.abs(X),
                2
                + 4 * (torch.arange(self.n_dim, device=self.device)) /
                (self.n_dim - 1),
            ),
            dim=-1,
        )
        return Y


class Rosenbrock:
    def __init__(self, n_dim: int, device=torch.device):
        self.n_dim = n_dim
        self.device = device

    def __call__(self, X):
        Y = 1e2 * torch.sum((X[:, :-1] ** 2 - X[:, 1:]) ** 2, dim=-1) + torch.sum(
            (X - 1) ** 2, dim=-1
        )
        return Y


class Schwefel:
    def __init__(self, n_dim: int, device=torch.device):
        self.n_dim = n_dim
        self.device = device
        self.C = torch.zeros(size=[self.n_dim, self.n_dim], device=self.device)
        for i in range(self.n_dim):
            self.C[i, :i] = 1.0
        self.C = torch.transpose(self.C, 0, 1)

    def __call__(self, X):
        PX = X**2  # [B, N]
        Y = torch.sum(PX @ self.C, dim=-1)
        return Y


class RotatedEllipsoid:
    def __init__(self, n_dim: int, device: torch.device):
        self.device = device
        self.n_dim = n_dim
        self.coefs = torch.pow(
            1e6, torch.arange(
                self.n_dim, device=self.device) / (self.n_dim - 1)
        )  # [self.n_dim]
        self.rot_mat, _ = torch.linalg.qr(
            torch.randn(self.n_dim, self.n_dim, device=self.device)
        )

    def __call__(self, X: torch.Tensor):
        X = torch.matmul(X, self.rot_mat)
        Y = torch.sum(self.coefs * (X**2), dim=-1)
        return Y


class RotatedBentCigar:
    def __init__(self, n_dim: int, device: torch.device):
        self.n_dim = n_dim
        self.device = device
        self.rot_mat, _ = torch.linalg.qr(
            torch.randn(self.n_dim, self.n_dim, device=self.device)
        )

    def __call__(self, X: torch.Tensor):
        # rot_mat [D, D], X [N, D]
        X = torch.matmul(X, self.rot_mat)  # [N, D]
        Y = X[:, 0] ** 2 + 1e6 * torch.sum(X[:, 1:] ** 2, dim=-1)
        return Y


class RotatedDiscus:
    def __init__(self, n_dim: int, device=torch.device):
        self.n_dim = n_dim
        self.device = device
        self.rot_mat, _ = torch.linalg.qr(
            torch.randn(self.n_dim, self.n_dim, device=self.device)
        )

    def __call__(self, X):
        X = torch.matmul(X, self.rot_mat)
        Y = 1e6 * X[:, 0] ** 2 + torch.sum(X[:, 1:] ** 2, dim=-1)
        return Y


class RotatedDiffPowers:
    def __init__(self, n_dim: int, device=torch.device):
        self.n_dim = n_dim
        self.device = device
        self.rot_mat, _ = torch.linalg.qr(
            torch.randn(self.n_dim, self.n_dim, device=self.device)
        )

    def __call__(self, X):
        X = torch.matmul(X, self.rot_mat)  # [N, D]
        Y = torch.sum(
            torch.pow(
                torch.abs(X),
                2
                + 4 * (torch.arange(self.n_dim, device=self.device)) /
                (self.n_dim - 1),
            ),
            dim=-1,
        )
        return Y


class RotatedRosenbrock:
    def __init__(self, n_dim: int, device=torch.device):
        self.n_dim = n_dim
        self.device = device
        self.rot_mat, _ = torch.linalg.qr(
            torch.randn(self.n_dim, self.n_dim, device=self.device)
        )

    def __call__(self, X):
        X = torch.matmul(X, self.rot_mat)  # [N, D]
        Y = 1e2 * torch.sum((X[:, :-1] ** 2 - X[:, 1:]) ** 2, dim=-1) + torch.sum(
            (X - 1) ** 2, dim=-1
        )
        return Y


class RotatedSchwefel:
    def __init__(self, n_dim: int, device=torch.device):
        self.n_dim = n_dim
        self.device = device
        self.C = torch.zeros(size=[self.n_dim, self.n_dim], device=self.device)
        for i in range(self.n_dim):
            self.C[i, :i] = 1.0
        self.C = torch.transpose(self.C, 0, 1)
        self.rot_mat, _ = torch.linalg.qr(
            torch.randn(self.n_dim, self.n_dim, device=self.device)
        )
        self.unrot_sc = Schwefel(n_dim, device)

    def __call__(self, X):
        X = torch.matmul(X, self.rot_mat)  # [N, D]
        # PX = X**2  # [B, N]
        # Y = PX @ self.C
        Y = self.unrot_sc(X)
        return Y


class MixedEllipsoid:
    def __init__(self, n_dim: int, device: torch.device):
        self.n_comps = 10
        self.sub_dim = n_dim // self.n_comps
        self.sub_rots = [
            torch.linalg.qr(
                torch.randn(self.sub_dim, self.sub_dim, device=device)
            )[0]
            for _ in range(self.n_comps)
        ]
        self.sub_fs = [BaseEllipsoid(self.sub_dim, device)
                       for _ in range(self.n_comps)]
        self.sub_idxs = torch.arange(n_dim).reshape([self.n_comps, -1])

    def __call__(self, X: torch.Tensor):
        Ys = torch.cat(
            [f(X[:, sub_idx] @ sub_rot) for f, sub_rot, sub_idx in zip(self.sub_fs, self.sub_rots, self.sub_idxs)], dim=-1
        )
        Y = Ys.sum(dim=-1)
        return Y


class MixedBentCigar:
    def __init__(self, n_dim: int, device: torch.device):
        self.n_comps = 10
        self.sub_dim = n_dim // self.n_comps
        self.sub_rots = [
            torch.linalg.qr(
                torch.randn(self.sub_dim, self.sub_dim, device=device)
            )[0]
            for _ in range(self.n_comps)
        ]
        self.sub_fs = [BaseBentCigar(self.sub_dim, device)
                       for _ in range(self.n_comps)]
        self.sub_idxs = torch.arange(n_dim).reshape([self.n_comps, -1])

    def __call__(self, X: torch.Tensor):
        Ys = torch.cat(
            [f(X[:, sub_idx] @ sub_rot) for f, sub_rot, sub_idx in zip(self.sub_fs, self.sub_rots, self.sub_idxs)], dim=-1
        )
        Y = Ys.sum(dim=-1)
        return Y


class MixedDiscus:
    def __init__(self, n_dim: int, device: torch.device):
        self.device = device
        self.n_dim = n_dim
        self.n_rot_dim = self.n_dim // 2
        self.n_sep_dim = self.n_dim // 2
        self.rot_mat, _ = torch.linalg.qr(
            torch.randn(self.n_rot_dim, self.n_rot_dim, device=self.device)
        )

    def __call__(self, X: torch.Tensor):
        X_rot, X_sep = X[:, : self.n_rot_dim], X[:, self.n_rot_dim:]
        X_rot = torch.matmul(X_rot, self.rot_mat)
        Y_rot = 1e6 * X_rot[:, 0] ** 2 + torch.sum(X_rot[:, 1:] ** 2, dim=-1)
        Y_sep = 1e6 * X_sep[:, 0] ** 2 + torch.sum(X_sep[:, 1:] ** 2, dim=-1)
        Y = Y_rot + Y_sep
        return Y


def get_func(rot, func, n_dim, device):
    if func == "Sphere":
        return Sphere(n_dim, device)
    elif func == "Ellipsoid":
        if rot == "Rotated":
            return RotatedEllipsoid(n_dim, device)
        elif rot == "Mixed":
            return MixedEllipsoid(n_dim, device)
        else:
            return Ellipsoid(n_dim, device)
    elif func == "BentCigar":
        if rot == "Rotated":
            return RotatedBentCigar(n_dim, device)
        elif rot == "Mixed":
            return MixedBentCigar(n_dim, device)
        else:
            return BentCigar(n_dim, device)
    elif func == "Discus":
        if rot == "Rotated":
            return RotatedDiscus(n_dim, device)
        elif rot == "Mixed":
            return MixedDiscus(n_dim, device)
        else:
            return Discus(n_dim, device)
    elif func == "Rosenbrock":
        if rot == "Rotated":
            return RotatedRosenbrock(n_dim, device)
        else:
            return Rosenbrock(n_dim, device)
    elif func == "DiffPowers":
        if rot == "Rotated":
            return RotatedDiffPowers(n_dim, device)
        else:
            return DiffPowers(n_dim, device)
    elif func == "Schwefel":
        if rot == "Rotated":
            return RotatedSchwefel(n_dim, device)
        else:
            return Schwefel(n_dim, device)
