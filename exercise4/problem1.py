import numpy as np
import matplotlib.pyplot as plt


# ADD CODE: read in the data here 
data = np.loadtxt("signal_data.txt")
t = data[:,0]
f = data[:,1]

dt = t[1]-t[0]
N=len(t)

# Fourier coefficients from numpy fft normalized by multiplication of dt
F = np.fft.fft(f)*dt

# frequencies from numpy fftfreq
freq = np.fft.fftfreq(N,d=dt)

# inverse Fourier with numpy ifft (normalization removed with division by dt)
iF = np.fft.ifft(F/dt)

# positive frequencies are given as
# freq[:N//2] from above or freq = np.linspace(0, 1.0/dt/2, N//2)
positive_freq = freq[:N//2]


fig, ax = plt.subplots()
# plot over positive frequencies the Fourier transform
#ax.plot(t, f, label='Original Signal')
ax.plot(freq[:N//2], np.abs(F[:N//2]))
ax.set_xlabel('$f$ (Hz)')
ax.set_ylabel('$F(\omega/2\pi)$')
ax.set_title("Part A: Frequency Spectrum of the Signal")

# PART B:
mask = (np.abs(freq) >= 40) & (np.abs(freq) <= 60)
F_filtered = F * mask

filtered_signal = np.fft.ifft(F_filtered / dt).real
# plot the "signal" and test the inverse transform
fig, ax = plt.subplots()
#ax.plot(t, f,t,iF.real,'r--')
ax.plot(t, f, label='Original', linestyle='-')
ax.plot(t, filtered_signal, 'r--', label='Filtered')
ax.set_xlabel('$t$ (s)')
ax.set_ylabel('$f(t)$')
ax.set_title('Problem B: Original and Filtered Signals in Time Domain')
plt.legend()
plt.show()


