import math
import numpy as np
from matplotlib import pyplot as plt

def xyzs_to_theta_phi(xyzs): # [..., 3]
  theta = np.arctan2(xyzs[..., 2], xyzs[..., 0])
  length = np.linalg.norm(xyzs, axis=-1)
  phi = np.arccos(xyzs[..., 1] / length)
  return np.stack([theta, phi], -1)

def cube_xys_to_equally_spaced_theta_phi(xys): # shape: [..., 2], range: [-1, 1]
  theta = xys[...,0] * math.pi / 4
  l = (np.tan(theta) ** 2 + 1) ** 0.5
  phi = np.arctan2(l, -xys[...,1])
  return np.stack([theta, phi], -1)

# 'The Equi-Angular Cubemap' sample method
if __name__ == '__main__':
  num = 10
  step = 2 / num
  offset = step / 2
  xs = np.linspace(-1 + offset, 1 - offset, num)
  xs, ys = np.meshgrid(xs, xs)
  xys = np.dstack([xs, ys])
  theta_phis = cube_xys_to_equally_spaced_theta_phi(xys)

  XS = np.sin(theta_phis[...,1]) * np.cos(theta_phis[...,0])
  YS = np.cos(theta_phis[...,1])
  ZS = np.sin(theta_phis[...,1]) * np.sin(theta_phis[...,0])
  XYZS1 = np.dstack([XS, YS, ZS])
  XYZS2 = np.dstack([-XS, YS, -ZS])
  XYZS3 = np.dstack([ZS, XS, YS])
  XYZS4 = np.dstack([ZS, -XS, -YS])
  XYZS5 = np.dstack([-ZS, YS, XS])
  XYZS6 = np.dstack([ZS, YS, -XS])

  a1 = xyzs_to_theta_phi(XYZS1)
  a2 = xyzs_to_theta_phi(XYZS2)
  a3 = xyzs_to_theta_phi(XYZS3)
  a4 = xyzs_to_theta_phi(XYZS4)
  a5 = xyzs_to_theta_phi(XYZS5)
  a6 = xyzs_to_theta_phi(XYZS6)

  plt.figure()
  plt.plot(a1[...,0].reshape([-1]), a1[...,1].reshape([-1]), 'g.')
  plt.plot(a2[...,0].reshape([-1]), a2[...,1].reshape([-1]), 'r.')
  plt.plot(a3[...,0].reshape([-1]), a3[...,1].reshape([-1]), 'b.')
  plt.plot(a4[...,0].reshape([-1]), a4[...,1].reshape([-1]), 'ro', mfc='none')
  plt.plot(a5[...,0].reshape([-1]), a5[...,1].reshape([-1]), 'go', mfc='none')
  plt.plot(a6[...,0].reshape([-1]), a6[...,1].reshape([-1]), 'bo', mfc='none')
  plt.xlim(-math.pi, math.pi)
  plt.ylim(0, math.pi)
  plt.show()
