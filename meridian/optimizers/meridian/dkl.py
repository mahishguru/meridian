"""Deep-Kernel GP surrogate with multi-task feasibility head for MERIDIAN.

A small MLP feature extractor ``phi_theta : R^d -> R^k`` is shared between
two heads:

  * **Regression head** — exact GP with ARD-Matern-5/2 kernel over phi(z),
    trained only on feasible observations. Provides calibrated (mu, sigma^2).
  * **Feasibility head** — logistic classifier over phi(z), trained on all
    observations (label = 1 if (decode + codec + DAMASK) succeeded).

Both heads share phi_theta so adding the classifier costs no extra network.
"""
from __future__ import annotations

from dataclasses import dataclass

import gpytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


class FeatureExtractor(nn.Module):
    """phi_theta : R^d -> R^k.  Small MLP with LayerNorm + GELU."""

    def __init__(self, in_dim: int, hidden: list[int], feature_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
            prev = h
        layers += [nn.Linear(prev, feature_dim), nn.LayerNorm(feature_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PropertyHeads(nn.Module):
    """Auxiliary heteroscedastic regression heads on top of phi(z).

    For each target property, two scalar outputs ``(mu_raw, log_var_raw)``
    are produced by a single linear layer over the shared feature
    embedding. Predictions are returned in **standardised** units; the
    surrogate maintains per-property ``(mean, std)`` statistics that are
    applied at predict time to recover physical units.

    The heads are trained jointly with the predictive proxy + feasibility
    classifier during the DKL pretraining phase, on rows where the
    corresponding property is finite (i.e. feasible simulations).
    """

    def __init__(self, feature_dim: int, n_props: int) -> None:
        super().__init__()
        # Single linear layer produces 2*n_props outputs (mu, log_var) per
        # property. Cheap; keeps the trunk dominant.
        self.head = nn.Linear(feature_dim, 2 * n_props)
        self.n_props = int(n_props)

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.head(feats)
        mu = out[..., : self.n_props]
        log_var = out[..., self.n_props :]
        # Clamp log_var for numerical stability (variance in [exp(-6), exp(4)])
        log_var = log_var.clamp(min=-6.0, max=4.0)
        return mu, log_var


@dataclass
class DKLState:
    """Container for the trained components after :meth:`DKLSurrogate.fit`."""

    feature_extractor: FeatureExtractor
    gp: SingleTaskGP | None
    classifier: nn.Linear
    y_mean: float
    y_std: float
    # ---- Optional auxiliary property heads (A2) ----
    property_heads: PropertyHeads | None = None
    property_keys: tuple[str, ...] = ()
    p_mean: np.ndarray | None = None  # (n_props,) physical-unit mean
    p_std: np.ndarray | None = None   # (n_props,) physical-unit std
    p_obs_mask: np.ndarray | None = None  # (n_props,) columns with labels


class DKLSurrogate:
    """DKL regressor + multi-task feasibility classifier.

    Re-fitted each round; the feature extractor is warm-started from the
    previous round's weights to amortise feature learning.

    Optionally also fits **auxiliary property heads** (A2): a small linear
    head per target property predicting heteroscedastic ``(mu, sigma)``
    over phi(z). The heads are activated by passing ``property_keys`` to
    :meth:`fit`; otherwise this class behaves exactly as before.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: list[int],
        feature_dim: int = 16,
        lr: float = 1e-3,
        epochs: int = 200,
        weight_decay: float = 1e-4,
        device: str = "cuda",
    ) -> None:
        self.in_dim = int(in_dim)
        self.hidden = list(hidden)
        self.feature_dim = int(feature_dim)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.weight_decay = float(weight_decay)
        self.device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.dtype = torch.double  # GP needs float64 for numerical stability

        # Persistent components (warm-start across rounds)
        self._phi: FeatureExtractor | None = None
        self._classifier: nn.Linear | None = None
        self._property_heads: PropertyHeads | None = None
        self.state: DKLState | None = None

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X_all: np.ndarray,
        Y_all: np.ndarray,
        feasible_mask: np.ndarray,
        P_all: np.ndarray | None = None,
        property_keys: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        """Fit the DKL regressor on feasible points and the classifier on all points.

        Parameters
        ----------
        X_all
            ``(N, d)`` all observed latents (feasible + infeasible).
        Y_all
            ``(N,)`` corresponding objective values; entries where
            ``feasible_mask`` is ``False`` are ignored by the regressor.
        feasible_mask
            ``(N,)`` boolean array marking successful evaluations.
        P_all
            Optional ``(N, n_props)`` array of measured properties, in the
            order of ``property_keys``. ``NaN`` rows are skipped during
            head training. When ``None``, no property heads are trained.
        property_keys
            Names of the properties in the columns of ``P_all``. Stored on
            the resulting :class:`DKLState` so consumers can map columns
            back to dict keys.
        """
        if self._phi is None:
            self._phi = FeatureExtractor(self.in_dim, self.hidden, self.feature_dim).to(
                self.device, dtype=self.dtype
            )
            self._classifier = nn.Linear(self.feature_dim, 1).to(
                self.device, dtype=self.dtype
            )

        # ---- Optional property heads (A2): lazily create / resize ----
        use_props = (
            P_all is not None and property_keys is not None and len(property_keys) > 0
        )
        if use_props:
            n_props = len(property_keys)
            if (
                self._property_heads is None
                or self._property_heads.n_props != n_props
            ):
                self._property_heads = PropertyHeads(self.feature_dim, n_props).to(
                    self.device, dtype=self.dtype
                )

        X_t = torch.from_numpy(X_all).to(self.device, dtype=self.dtype)
        feas_t = torch.from_numpy(feasible_mask.astype(np.float32)).to(
            self.device, dtype=self.dtype
        )
        feas_idx = torch.where(feas_t > 0.5)[0]

        # Standardise feasible y for GP
        y_feas = Y_all[feasible_mask]
        if len(y_feas) >= 2:
            y_mean = float(y_feas.mean())
            y_std = float(y_feas.std() + 1e-6)
        else:
            y_mean, y_std = 0.0, 1.0

        # ---- 1. Joint pretraining of phi_theta + classifier (warm-started) ----
        # Use a cheap predictive proxy on phi (linear regressor) so phi learns
        # features useful for *both* regression and feasibility classification
        # before handing phi over to the exact GP. When property heads are
        # active, their Gaussian-NLL loss is added to the joint objective so
        # phi additionally learns property-discriminative features.
        p_mean_arr: np.ndarray | None = None
        p_std_arr: np.ndarray | None = None
        p_obs_mask_arr: np.ndarray | None = None
        if use_props:
            # Standardise per-property over feasible+finite labels. Cached
            # seed manifests may omit ``n_grains`` (or store -1), while still
            # providing valid mechanical labels; train each head column on the
            # labels it actually has instead of requiring complete rows.
            p_mat = np.asarray(P_all, dtype=np.float64)
            finite_p = np.isfinite(p_mat) & feasible_mask[:, None]
            counts = finite_p.sum(axis=0)
            p_obs_mask_arr = counts >= 2
            if bool(p_obs_mask_arr.any()):
                n_props = p_mat.shape[1]
                p_mean_arr = np.zeros(n_props, dtype=np.float64)
                p_std_arr = np.ones(n_props, dtype=np.float64)
                p_std_full = np.zeros_like(p_mat, dtype=np.float64)
                for j, has_labels in enumerate(p_obs_mask_arr):
                    if not bool(has_labels):
                        continue
                    vals = p_mat[finite_p[:, j], j]
                    p_mean_arr[j] = float(vals.mean())
                    p_std_arr[j] = float(vals.std() + 1e-6)
                    p_std_full[finite_p[:, j], j] = (
                        p_mat[finite_p[:, j], j] - p_mean_arr[j]
                    ) / p_std_arr[j]
                valid_p_rows = finite_p[:, p_obs_mask_arr].any(axis=1)
                p_std_t = torch.from_numpy(
                    p_std_full[valid_p_rows].astype(np.float32)
                ).to(self.device, dtype=self.dtype)
                p_mask_t = torch.from_numpy(
                    (finite_p[valid_p_rows] & p_obs_mask_arr[None, :]).astype(np.float32)
                ).to(self.device, dtype=self.dtype)
                valid_p_idx = torch.from_numpy(np.where(valid_p_rows)[0]).to(self.device)
            else:
                # Not enough data yet — disable for this round
                use_props = False

        if len(feas_idx) >= 2:
            y_feas_t = torch.from_numpy(
                ((Y_all[feasible_mask] - y_mean) / y_std).astype(np.float32)
            ).to(self.device, dtype=self.dtype)
            # Shared linear regression head used only during pretraining
            reg_head = nn.Linear(self.feature_dim, 1).to(self.device, dtype=self.dtype)
            params = (
                list(self._phi.parameters())
                + list(self._classifier.parameters())
                + list(reg_head.parameters())
            )
            if use_props:
                params += list(self._property_heads.parameters())
            opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
            for _ in range(self.epochs):
                opt.zero_grad()
                z_feat = self._phi(X_t)
                logit = self._classifier(z_feat).squeeze(-1)
                cls_loss = F.binary_cross_entropy_with_logits(logit, feas_t)
                pred_feas = reg_head(z_feat[feas_idx]).squeeze(-1)
                reg_loss = F.mse_loss(pred_feas, y_feas_t)
                loss = reg_loss + 0.1 * cls_loss
                if use_props:
                    feats_p = z_feat[valid_p_idx]
                    mu_p, log_var_p = self._property_heads(feats_p)
                    # Gaussian heteroscedastic NLL (standardised units)
                    diff = mu_p - p_std_t
                    nll = 0.5 * (log_var_p + diff.pow(2) * torch.exp(-log_var_p))
                    prop_loss = (nll * p_mask_t).sum() / p_mask_t.sum().clamp_min(1.0)
                    # Weight 0.5 keeps property heads from dominating phi
                    loss = loss + 0.5 * prop_loss
                loss.backward()
                opt.step()
        else:
            # Only feasibility loss when no regression data
            opt = torch.optim.AdamW(
                list(self._phi.parameters()) + list(self._classifier.parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            for _ in range(self.epochs // 2):
                opt.zero_grad()
                logit = self._classifier(self._phi(X_t)).squeeze(-1)
                F.binary_cross_entropy_with_logits(logit, feas_t).backward()
                opt.step()

        # ---- 2. Exact GP on the (now meaningful) features --------------------
        gp = None
        if len(feas_idx) >= 2:
            with torch.no_grad():
                feats_feas = self._phi(X_t[feas_idx])
            y_feas_std = torch.from_numpy(
                ((Y_all[feasible_mask] - y_mean) / y_std).astype(np.float32)
            ).to(self.device, dtype=self.dtype)
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-2))
            covar = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=self.feature_dim,
                    lengthscale_constraint=Interval(0.005, 4.0),
                )
            )
            gp = SingleTaskGP(
                feats_feas, y_feas_std.unsqueeze(-1),
                likelihood=likelihood, covar_module=covar,
            ).to(self.device, dtype=self.dtype)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                pass

        self.state = DKLState(
            feature_extractor=self._phi,
            gp=gp,
            classifier=self._classifier,
            y_mean=y_mean,
            y_std=y_std,
            property_heads=self._property_heads if use_props else None,
            property_keys=tuple(property_keys) if use_props else (),
            p_mean=p_mean_arr if use_props else None,
            p_std=p_std_arr if use_props else None,
            p_obs_mask=p_obs_mask_arr if use_props else None,
        )

    # -------------------------------------------------------------- predict
    @torch.no_grad()
    def features(self, X: np.ndarray) -> torch.Tensor:
        X_t = torch.from_numpy(X).to(self.device, dtype=self.dtype)
        return self.state.feature_extractor(X_t)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior (mean, std) on the *original* y scale."""
        feats = self.features(X)
        post = self.state.gp.posterior(feats)
        mu = post.mean.squeeze(-1).cpu().numpy() * self.state.y_std + self.state.y_mean
        sigma = post.variance.squeeze(-1).clamp_min(0).sqrt().cpu().numpy() * self.state.y_std
        return mu, sigma

    @torch.no_grad()
    def feasibility(self, X: np.ndarray) -> np.ndarray:
        feats = self.features(X)
        return torch.sigmoid(self.state.classifier(feats).squeeze(-1)).cpu().numpy()

    @torch.no_grad()
    def predict_properties(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Predict ``(mu, sigma)`` for each auxiliary property.

        Returns
        -------
        ``(mu, sigma)`` of shape ``(N, n_props)`` in *physical* units,
        ordered to match :attr:`DKLState.property_keys`. Returns ``None``
        if property heads have not been fitted yet.
        """
        if (
            self.state is None
            or self.state.property_heads is None
            or self.state.p_mean is None
            or self.state.p_std is None
        ):
            return None
        feats = self.features(X)
        mu_std, log_var = self.state.property_heads(feats)
        sigma_std = (0.5 * log_var).exp()
        # De-standardise back to physical units
        p_mean = torch.from_numpy(self.state.p_mean).to(self.device, dtype=self.dtype)
        p_std = torch.from_numpy(self.state.p_std).to(self.device, dtype=self.dtype)
        mu = (mu_std * p_std + p_mean).cpu().numpy()
        sigma = (sigma_std * p_std).cpu().numpy()
        return mu, sigma

    # --------------------------------------------------- A5: head reset
    def reset_property_heads(self) -> None:
        """Re-initialise PropertyHeads linear weights, keep trunk (A5).

        Called periodically by MERIDIAN to prevent the heads from
        overfitting on a saturated cache (sigma -> floor everywhere). The
        shared feature extractor and feasibility classifier are left
        untouched so the surrogate retains learned representations.
        """
        if self._property_heads is None:
            return
        n_props = int(self._property_heads.n_props)
        self._property_heads = PropertyHeads(self.feature_dim, n_props).to(
            self.device, dtype=self.dtype
        )

    # --------------------------------------------------- A4: differentiable predict
    def predict_properties_diff(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Differentiable property mean/std in *physical* units.

        Mirrors :meth:`predict_properties` but keeps the autograd graph so
        gradients can flow back to ``z``. Used by A4 (gradient seeding)
        to do inverse design directly on
        ``L(z) = sum_i w_i ((mu_p_i(z) - target_i)/|target_i|)^2``.

        Returns ``None`` if heads are not yet fitted.
        """
        if (
            self.state is None
            or self.state.property_heads is None
            or self.state.p_mean is None
            or self.state.p_std is None
        ):
            return None
        z = z.to(self.device, dtype=self.dtype)
        feats = self.state.feature_extractor(z)
        mu_std, log_var = self.state.property_heads(feats)
        sigma_std = (0.5 * log_var).exp()
        p_mean = torch.from_numpy(self.state.p_mean).to(self.device, dtype=self.dtype)
        p_std = torch.from_numpy(self.state.p_std).to(self.device, dtype=self.dtype)
        mu = mu_std * p_std + p_mean
        sigma = sigma_std * p_std
        return mu, sigma

    # --------------------------------------------------- A4: differentiable feasibility
    def feasibility_diff(self, z: torch.Tensor) -> torch.Tensor:
        """Differentiable feasibility probability for projection in A4."""
        z = z.to(self.device, dtype=self.dtype)
        feats = self.state.feature_extractor(z)
        return torch.sigmoid(self.state.classifier(feats).squeeze(-1))

    # --------------------------------------------------- gradient for active subspace
    def grad_mean(self, X: np.ndarray) -> np.ndarray:
        """``d mu / d z`` for the standardised posterior mean.

        Returns a ``(N, d)`` array of gradients in the *original* z-space.
        Used by :class:`SubspaceManager` for active-subspace estimation.
        """
        X_t = torch.from_numpy(X).to(self.device, dtype=self.dtype).requires_grad_(True)
        feats = self.state.feature_extractor(X_t)
        # Use the GP's posterior mean as a differentiable function of phi(z)
        with gpytorch.settings.fast_pred_var():
            mu = self.state.gp.posterior(feats).mean.squeeze(-1)
        grads = torch.autograd.grad(mu.sum(), X_t)[0]
        return grads.detach().cpu().numpy()
