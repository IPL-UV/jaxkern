from typing import Optional, Tuple
import jax
import jax.numpy as np
import objax
from objax.typing import JaxArray
from jaxkern.gp.predictive import predictive_mean


class UnscentedTransform(objax.Module):
    def __init__(
        self,
        model,
        kappa: Optional[float] = None,
        alpha: float = 1.0,
        beta: float = 2.0,
        jitted: bool = False,
        seed: int = 123,
    ):
        self.n_features = model.X_train_.shape[-1]

        # generate sigma weights
        wm, wc = get_unscented_weights(
            self.n_features, kappa=kappa, alpha=alpha, beta=beta
        )
        self.wm = wm.squeeze()
        self.wc = np.diag(wc.squeeze())

        # generate sigma points
        self.sigma_points = get_unscented_sigma_points(
            self.n_features, kappa=kappa, alpha=alpha
        )

        f = jax.vmap(jax.partial(predictive_mean, model))

        transform = jax.vmap(
            jax.partial(_transform, f), in_axes=(0, 0, None, None, None)
        )

        if jitted:
            transform = jax.jit(transform)

        self.transform = transform

    def forward(self, X, Xcov) -> Tuple[JaxArray, JaxArray]:

        # form sigma points from unit sigma-points
        mean_f, var_f = self.transform(
            X,
            Xcov,
            self.sigma_points,
            self.wm,
            self.wc,
        )

        return mean_f, var_f


def _transform(mean_f, X, Xcov, sigma_points, wm, wc):

    # form sigma points from unit sigma-points
    # print(Xcov.shape, Xcov.shape, sigma_points.shape)
    x_ = X[:, None] + np.linalg.cholesky(Xcov) @ sigma_points
    # print("x_:", x_.shape)

    fx_ = mean_f(x_.T)
    # print("fx_:", fx_.shape, ", wm_:", wm.shape)

    # output mean
    mean_f = np.sum(fx_ * wm)
    # print("mean_f:", mean_f.shape)

    # output covariance
    dfx_ = (fx_ - mean_f)[:, None]
    # print(dfx_.shape)

    cov_f = dfx_.T @ wc @ dfx_
    # print("Cov_f:", cov_f.shape)
    return mean_f, np.diag(cov_f)


def get_unscented_weights(
    n_features: int,
    kappa: Optional[float] = None,
    alpha: float = 1.0,
    beta: float = 2.0,
) -> Tuple[float, float]:
    """Generate normalizers for MCMC samples"""

    # calculate kappa value
    if kappa is None:
        kappa = np.maximum(3.0 - n_features, 0.0)

    lam = alpha ** 2 * (n_features + kappa) - n_features
    wm = 1.0 / (2.0 * (n_features + lam)) * np.ones(2 * n_features + 1)
    wc = wm.copy()
    wm = jax.ops.index_update(wm, 0, lam / (n_features + lam))
    wc = jax.ops.index_update(wc, 0, wm[0] + (1 - alpha ** 2 + beta))
    return wm, wc


def get_unscented_sigma_points(
    n_features: int, kappa: Optional[float] = None, alpha: float = 1.0
) -> JaxArray:
    """Generate Unscented samples"""

    # calculate kappa value
    if kappa is None:
        kappa = np.maximum(3.0 - n_features, 0.0)

    lam = alpha ** 2 * (n_features + kappa) - n_features
    c = np.sqrt(n_features + lam)
    return np.hstack(
        (np.zeros((n_features, 1)), c * np.eye(n_features), -c * np.eye(n_features))
    )