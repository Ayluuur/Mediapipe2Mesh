import numpy as np


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

    last_update = 0
    last_mse = 0
    # Do not modify the caller's warm-start buffer in-place.
    params = np.array(init, dtype=np.float64, copy=True)
    for i in range(self.max_iter):
      baseline = model.run(params)
      residual = (baseline - target).reshape(out_n, 1)
      mse = np.mean(np.square(residual))

      if abs(mse - last_mse) < self.mse_threshold:
        return params

      for k in range(params.shape[0]):
        jacobian[:, k] = self.get_derivative(
          model, params, k, baseline=baseline
        )

      jtj = np.matmul(jacobian.T, jacobian)
      jtj = jtj + (u + regularization) * np.eye(jtj.shape[0])

      update = last_mse - mse
      rhs = np.matmul(jacobian.T, residual)
      if regularization:
        rhs += regularization * (params - prior).reshape(-1, 1)
      try:
        delta = np.linalg.solve(jtj, rhs).ravel()
      except np.linalg.LinAlgError:
        delta = np.linalg.lstsq(jtj, rhs, rcond=None)[0].ravel()
      params -= delta

      if update > last_update and update > 0:
        u /= v
      else:
        u *= v

      last_update = update
      last_mse = mse

    return params
