import numpy as np

### Function for decomposing signal into brain wave frequency bands ###

def decompose(signal , sampling_rate , n_samples):
  #####
  #input: eeg data , sampling rate , and number of samples
  #output: list containing eeg signal decomposed into the 5 brain wave frequency bands

  fourier_signal = np.fft.fft(signal)
  frequencies = np.fft.fftfreq(n_samples , 1/sampling_rate)
  nyquist_limit = sampling_rate / 2

  gamma = np.zeros((fourier_signal.shape) , dtype = complex)
  beta = np.zeros((fourier_signal.shape) , dtype = complex)
  alpha = np.zeros((fourier_signal.shape) , dtype = complex)
  theta = np.zeros((fourier_signal.shape) , dtype = complex)
  delta = np.zeros((fourier_signal.shape) , dtype = complex)

  Gmask = (abs(frequencies) >= 30) & (abs(frequencies) <= nyquist_limit)
  Bmask = (abs(frequencies) >= 12) & (abs(frequencies) < 30)
  Amask = (abs(frequencies) >= 8) & (abs(frequencies) < 12)
  Tmask = (abs(frequencies) >= 4) & (abs(frequencies) < 8)
  Dmask = (abs(frequencies) >= 0.5) & (abs(frequencies) < 4)

  gamma[Gmask] = fourier_signal[Gmask]
  beta[Bmask] = fourier_signal[Bmask]
  alpha[Amask] = fourier_signal[Amask]
  theta[Tmask] = fourier_signal[Tmask]
  delta[Dmask] = fourier_signal[Dmask]

  return [np.fft.ifft(gamma) , np.fft.ifft(beta) , np.fft.ifft(alpha) , np.fft.ifft(theta) , np.fft.ifft(delta)]
