import numpy as np
from scipy.linalg import cho_factor, cho_solve


class Solver:
  def __init__(self, eps=1e-5, max_iter=30, mse_threshold=1e-8,
               verbose=False, central_difference=True):
    """
    Parameters
    ----------
    eps : float, optional
      Epsilon for derivative computation, by default 1e-5
    max_iter : int, optional
      Max iterations, by default 30
    mse_threshold : float, optional
      Early top when mse change is smaller than this threshold, by default 1e-8
    verbose : bool, optional
      Print information in each iteration, by default False
    central_difference : bool, optional
      Use the more accurate central finite difference. When False, use a
      forward difference which needs roughly half as many model evaluations.
    """
    self.eps = eps
    self.max_iter = max_iter
    self.mse_threshold = mse_threshold
    self.verbose = verbose
    self.central_difference = central_difference

  def get_derivative(self, model, params, n, baseline=None):
    """
    Compute the derivative by adding and subtracting epsilon

    Parameters
    ----------
    model : object
      Model wrapper to be manipulated.
    params : np.ndarray
      Current model parameters.
    n : int
      The index of parameter.

    Returns
    -------
    np.ndarray
      Derivative with respect to the n-th parameter.
    """
    params1 = np.array(params, copy=True)
    params1[n] += self.eps
    res1 = model.run(params1)

    if self.central_difference:
      params2 = np.array(params, copy=True)
      params2[n] -= self.eps
      res2 = model.run(params2)
      d = (res1 - res2) / (2 * self.eps)
    else:
      if baseline is None:
        baseline = model.run(params)
      d = (res1 - baseline) / self.eps

    return d.ravel()

  def solve(self, model, target, init=None, u=1e-3, v=1.5,
            prior=None, regularization=0.0):
    """
    Solver for the target.

    Parameters
    ----------
    model : object
      Wrapper to be manipulated.
    target : np.ndarray
      Optimization target.
    init : np,ndarray, optional
      Initial parameters, by default None
    u : float, optional
      LM algorithm parameter, by default 1e-3
    v : float, optional
      LM algorithm parameter, by default 1.5
    prior : np.ndarray, optional
      Pose prior used by the regularization term, by default None
    regularization : float, optional
      L2 weight which keeps parameters close to ``prior``, by default 0

    Returns
    -------
    np.ndarray
      Solved model parameters.
    """
    if init is None:
      init = np.zeros(model.n_params)
    if prior is None:
      prior = np.zeros(model.n_params)
    prior = np.asarray(prior, dtype=np.float64)
    out_n = np.shape(model.run(init).ravel())[0]
    jacobian = np.zeros([out_n, init.shape[0]])

    # Do not modify the caller's warm-start buffer in-place.
    params = np.array(init, dtype=np.float64, copy=True)
    u = max(float(u), 1e-12)
    if v <= 1 or regularization < 0:
      raise ValueError('v must exceed 1 and regularization must be nonnegative')
    for i in range(self.max_iter):
      baseline = model.run(params)
      residual = (baseline - target).reshape(out_n, 1)
      cost = np.sum(residual ** 2) + regularization * np.sum((params - prior) ** 2)

      if callable(getattr(model, 'run_batch', None)):
        perturbations = params + self.eps * np.eye(params.size)
        if self.central_difference:
          samples = model.run_batch(np.concatenate((
            perturbations, params - self.eps * np.eye(params.size)
          )))
          derivatives = (samples[:params.size] - samples[params.size:]) / (2 * self.eps)
        else:
          derivatives = (model.run_batch(perturbations) - baseline) / self.eps
        jacobian[:] = derivatives.reshape(params.size, out_n).T
      else:
        for k in range(params.shape[0]):
          jacobian[:, k] = self.get_derivative(
            model, params, k, baseline=baseline
          )

      jtj = np.matmul(jacobian.T, jacobian)
      rhs = np.matmul(jacobian.T, residual)
      if regularization:
        rhs += regularization * (params - prior).reshape(-1, 1)
      accepted = False
      improvement = 0.0
      for _ in range(32):
        system = jtj + (u + regularization) * np.eye(jtj.shape[0])
        try:
          delta = cho_solve(cho_factor(system, check_finite=False),
                            rhs, check_finite=False).ravel()
        except np.linalg.LinAlgError:
          delta = np.linalg.lstsq(system, rhs, rcond=None)[0].ravel()
        candidate = params - delta
        result = model.run(candidate)
        candidate_cost = (np.sum((result - target) ** 2)
                          + regularization * np.sum((candidate - prior) ** 2))
        if np.isfinite(candidate_cost) and candidate_cost < cost:
          params = candidate
          improvement = cost - candidate_cost
          u = max(u / v, 1e-12)
          accepted = True
          break
        u *= v
      if not accepted or improvement / out_n < self.mse_threshold:
        break
    # Derivatives and rejected trials also mutate the wrapped model.
    model.run(params)
    return params
