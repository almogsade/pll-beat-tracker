import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal


def onset_chunk_generator(chunk_generator, half_window):

    window = np.hanning(2*half_window + 1)
    prev_energy = 0.0
    first = True

    for chunk in chunk_generator:

        if chunk.ndim > 1:
            chunk = np.mean(chunk, axis=1)

        if first:
            chunk = np.concatenate((np.zeros(half_window, dtype=chunk.dtype), chunk))
            first = False

        energy = np.convolve(chunk**2, window, mode='valid')

        onset = np.maximum(np.diff(energy, prepend=prev_energy), 0)
        prev_energy = energy[-1]
        onset -= np.average(onset)

        yield onset

class ArrayBuffer:
    def __init__(self, max_size):
        self.buffer = np.zeros(max_size, dtype=np.float32)
        self.max_size = max_size
        self.index = 0
        self.count = 0

    def append(self, value):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.max_size
        self.count = min(self.count + 1, self.max_size)

    def get_last(self, N):
        N = int(min(N, self.count))
        # Compute indices for the last N samples
        idx = (self.index - N + np.arange(N)) % self.max_size
        return self.buffer[idx]

class RealTimePLL:
    def __init__(self, sr, freq, gain, max_mem_s):
        self.sr = sr                      # samples per second
        self.freq = freq                  # Hz
        self.per_sample = freq / sr       
        self.gain = gain
        
        max_size = int(max_mem_s * sr)
        self.order_0_buffer = ArrayBuffer(max_size)
        self.phase = 0.0
        self.order_1 = 0.0

    def update(self, onset_value):
        inst_per_sample = self.per_sample + self.gain * self.order_1
        self.phase = (self.phase + 2 * np.pi * inst_per_sample) % (2 * np.pi)

        x_t = np.sin(self.phase)
        self.order_0_buffer.append(x_t * onset_value)

        N = int(min(max(1 / max(inst_per_sample, 1e-9), 1), self.order_0_buffer.max_size))
        self.order_1 = np.sum(self.order_0_buffer.get_last(N)) / N

        return x_t, inst_per_sample * sr


file_path = 'TAAT_trimmed_15s.wav'
sr = sf.SoundFile(file_path).samplerate

chunk_size = 1024
half_window = 256

chunk_generator = sf.blocks(file_path, blocksize=chunk_size, overlap=half_window, dtype='float32')

# Create onset stream generator
onset_chunks = onset_chunk_generator(chunk_generator, half_window)

# Initialize PLL
pll = RealTimePLL(sr=sr, freq=2.5, gain=0.02, max_mem_s=1.0)

pll_output = []
inst_freq = []
onset_signal = []

for onset_chunk in onset_chunks:
    for onset_value in onset_chunk:
        #x, f = pll.update(onset_value)
        #pll_output.append(x)
        #inst_freq.append(f)
        onset_signal.append(onset_value)



# === 4. Normalize for better visualization ===
# pll_output /= np.max(np.abs(pll_output))
#onset_signal /= np.max(np.abs(onset_signal))

# === 5. Time vector ===
t = np.arange(len(onset_signal)) / sr

# === 6. Reference sine ===
#ref_freq = pll.freq
#ref_sine = np.sin(2 * np.pi * ref_freq * t)

# === 7. Plot ===
plt.figure(figsize=(12, 6))
plt.subplot(2,1,1)
plt.plot(t, onset_signal, label='Onset Signal', alpha=0.6)
#plt.plot(t, pll_output, label='PLL VCO Output', alpha=0.8)
#plt.plot(t, ref_sine, '--', label='Reference Sine (Nominal Freq)', alpha=0.5)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude (normalized)')
plt.title('PLL Locking Behavior with Reference Frequency')
plt.legend()

plt.subplot(2,1,2)
#plt.plot(t, inst_freq, label='instant frequency', alpha=0.8)
plt.xlabel('Time [s]')
plt.ylabel('frequency')
plt.legend()


plt.grid(True)
plt.tight_layout()
plt.show()


