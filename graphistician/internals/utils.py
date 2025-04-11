from typing import Any, Union

import numpy as np
import scipy

from scipy.special import erfcinv


def logit(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert a probability value (or array of probability values) to log-odds.
    
    Args:
        p (float or array-like): Probability in the range [0, 1].
    
    Returns:
        float or array-like: The log-odds corresponding to the given probability(ies).
    """
    return scipy.special.logit(p)


def normal_inverse_wishart_log_prob(gaussian: Any) -> float:
    """
    Compute the log probability under a normal-inverse-Wishart (NIW) prior for a Gaussian object.
    
    This function uses both an inverse-Wishart distribution (for the covariance) and 
    a multivariate normal distribution (for the mean) to compute the overall log probability.

    Args:
        gaussian (object): A Gaussian-like object containing:
            sigma (array-like): Covariance matrix.
            nu_0 (float): Degrees of freedom for inverse-Wishart.
            sigma_0 (array-like): Scale matrix for the inverse-Wishart distribution.
            mu (array-like): Mean vector of the Gaussian.
            mu_0 (array-like): Prior mean vector.
            kappa_0 (float): Prior precision scaling factor for the mean.

    Returns:
        float: Log probability of the given Gaussian under the NIW prior.
    """
    from scipy.stats import invwishart, multivariate_normal
    lp = 0.0

    # Log probability of the covariance under the inverse-Wishart
    lp += invwishart.logpdf(
        gaussian.sigma,
        df=gaussian.nu_0,
        scale=gaussian.sigma_0
    )

    # Log probability of the mean under the normal distribution with scaled covariance
    lp += multivariate_normal.logpdf(
        gaussian.mu,
        mean=gaussian.mu_0,
        cov=gaussian.sigma / gaussian.kappa_0
    )

    return lp

def sample_truncnorm(
    mu: Union[float, np.ndarray] = 0,
    sigma: Union[float, np.ndarray] = 1,
    lb: Union[float, np.ndarray] = -np.inf,
    ub: Union[float, np.ndarray] = np.inf
) -> np.ndarray:
    """
    Draw samples from a truncated normal distribution, broadcasting input parameters as needed.
    
    Args:
        mu (float or array-like, optional): Mean(s) of the normal distribution. Defaults to 0.
        sigma (float or array-like, optional): Standard deviation(s). Defaults to 1.
        lb (float or array-like, optional): Lower bound(s) of truncation. Defaults to -np.inf.
        ub (float or array-like, optional): Upper bound(s) of truncation. Defaults to np.inf.

    Returns:
        numpy.ndarray: Samples drawn from the specified truncated normal distribution.
    """
    mu, sigma, lb, ub = np.broadcast_arrays(mu, sigma, lb, ub)
    shp = mu.shape

    if np.allclose(sigma, 0.0):
        return mu

    cdflb = scipy.stats.norm.cdf(lb, loc=mu, scale=sigma)
    cdfub = scipy.stats.norm.cdf(ub, loc=mu, scale=sigma)

    cdfsamples = cdflb + np.random.rand(*shp) * (cdfub - cdflb)
    cdfsamples = np.clip(cdfsamples, 1e-15, 1 - 1e-15)

    zs = -np.sqrt(2) * erfcinv(2 * cdfsamples)
    assert np.all(np.isfinite(zs)), "Truncated normal samples should be finite."

    return sigma * zs + mu

def expected_truncnorm(
    mu: Union[float, np.ndarray] = 0,
    sigma: Union[float, np.ndarray] = 1.0,
    lb: Union[float, np.ndarray] = -np.inf,
    ub: Union[float, np.ndarray] = np.inf
) -> np.ndarray:
    """
    Compute the expected value of a truncated normal random variable.

    Args:
        mu (float or array-like, optional): Mean(s) of the normal distribution. Defaults to 0.
        sigma (float or array-like, optional): Standard deviation(s). Defaults to 1.
        lb (float or array-like, optional): Lower bound(s) of truncation. Defaults to -np.inf.
        ub (float or array-like, optional): Upper bound(s) of truncation. Defaults to np.inf.

    Returns:
        numpy.ndarray: Expected value of the truncated normal distribution.
    
    Notes:
    - This function uses the form and notation from Wikipedia:
        http://en.wikipedia.org/wiki/Truncated_normal_distribution
    - The function is designed to handle broadcasting over arrays of mu, sigma, lb, and ub.
    - The function will return the expected value of the truncated normal distribution
    - The only reason we don't use the scipy version is that we want to
      broadcast over arrays of mu, sigma, lb, and ub.
    """
    # Broadcast arrays to be of the same shape
    mu, sigma, lb, ub = np.broadcast_arrays(mu, sigma, lb, ub)
    if np.allclose(sigma, 0.0):
        return mu

    # Compute the normalizer of the truncated normal
    Z = scipy.stats.norm.cdf(ub) - scipy.stats.norm.cdf(lb)
    # Standardize the bounds
    alpha = (lb - mu) / sigma
    beta = (ub - mu) / sigma

    E = mu + (scipy.stats.norm.pdf(alpha) - scipy.stats.norm.pdf(beta)) / Z * sigma
    return E

def compute_optimal_rotation(
    L: np.ndarray,
    L_true: np.ndarray,
    scale: bool = True
) -> np.ndarray:
    """
    Find a rotation matrix R such that F_inf.dot(R) ~= F_true.

    Args:
        L (np.ndarray): Input matrix to be rotated.
        L_true (np.ndarray): Target matrix for alignment.
        scale (bool, optional): Whether to scale the rotation matrix. Defaults to True.

    Returns:
        np.ndarray: Optimal rotation matrix.
    """
    from scipy.linalg import orthogonal_procrustes
    R = orthogonal_procrustes(L, L_true)[0]

    # Given R, find a scale such that Lp' = Lp*s is such that,
    # s = argmin_s y = (Lp * s - L)^T (Lp*s - L)
    #     d/ds y = 2 * (Lp * s - L)^T Lp = 0
    #            =>     Lp^T Lp *s - L^T Lp = 0
    #            =>     s = (L^T Lp) / (Lp^T Lp)
    #
    # Then roll s into R such that Lp' = Lp*s = L.dot(R)*s = L.dot(R')
    # for R' = R*s

    if scale:
        Lp = L.dot(R)
        s = (L_true * Lp).sum() / (Lp * Lp).sum()
        return R * s
    else:
        return R
