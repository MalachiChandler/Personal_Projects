### My own DFT , IDFT , and cleanup functions ###
import numpy as np
def DFT(x):
  N = len(x)
  f = []
  for j in range(N):
    fj = 0
    for k in range(N):
      fj += x[k] / (np.exp( 2j * np.pi * k * j * (1/N) ))
    f.append(fj)
  return np.array(f)

def inverse_DFT(x):
  N = len(x)
  f = []
  for j in range(N):
    fj = 0
    for k in range(N):
      fj += x[k] * (np.exp( 2j * np.pi * k * j * (1/N) ))
    f.append(fj)
  return np.array(f)

def DFT_cleanup(DFT_signal , lower_bound=1e-8):
  f = DFT_signal
  F = np.zeros(len(f)//2 + 1)
  for i in range(len(F)):
    m = np.sqrt( 2 * ( f[i].real**2 + f[i].imag**2 ) ) / len(f)
    if m < lower_bound:
      F[i] = 0
    else:
      F[i] = m
  return np.array(F)
