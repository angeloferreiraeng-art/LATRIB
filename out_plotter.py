import csv
import matplotlib.pyplot as plt
import numpy as np


PATH = r"../dados_angelo/rec_b3.csv"


def plot_fft_spectrum(signal, fs=1.0, n_peaks=5):
    """
    Compute and plot the FFT spectrum of a signal, annotating the largest peaks.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array).
    fs : float
        Sampling frequency (Hz). Defaults to 1.0 (normalized frequency).
    n_peaks : int
        Number of largest peaks to annotate.

    Returns
    -------
    freqs : np.ndarray
        Frequency bins.
    magnitude : np.ndarray
        Magnitude spectrum (one-sided).
    peak_freqs : np.ndarray
        Frequencies of the top peaks.
    peak_mags : np.ndarray
        Magnitudes of the top peaks.
    """
    signal = np.asarray(signal).flatten()
    N = len(signal)

    # Compute FFT and one-sided spectrum
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    magnitude = np.abs(fft_vals) / N  # normalize
    magnitude[1:-1] *= 2  # account for two-sided -> one-sided energy

    # Find the top n_peaks local maxima
    peaks = []
    for i in range(1, len(magnitude) - 1):
        if magnitude[i] > magnitude[i - 1] and magnitude[i] > magnitude[i + 1]:
            peaks.append((magnitude[i], freqs[i]))
    peaks.sort(reverse=True)

    # Fall back to global top if not enough local maxima found
    if len(peaks) < n_peaks:
        idx = np.argsort(magnitude)[::-1][:n_peaks]
        peaks = [(magnitude[i], freqs[i]) for i in idx]
    else:
        peaks = peaks[:n_peaks]

    peak_mags = np.array([p[0] for p in peaks])
    peak_freqs = np.array([p[1] for p in peaks])

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, magnitude, color="steelblue", lw=1)
    plt.scatter(peak_freqs, peak_mags, color="red", zorder=5)

    for f, m in zip(peak_freqs, peak_mags):
        plt.annotate(f"{f:.2f} Hz\n({m:.3g})",
                     xy=(f, m),
                     xytext=(5, 8),
                     textcoords="offset points",
                     fontsize=9,
                     color="darkred",
                     arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

    plt.title(f"FFT Spectrum — top {n_peaks} peaks")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return freqs, magnitude, peak_freqs, peak_mags


with open(PATH, mode="r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    print(header)
    cell_data = []
    pos_data = []
    for i in range(400):
        next(reader)
    for row in reader:
        # print(type(row[0]), type(row[1]))
        cell_data.append(float(row[0]))
        pos_data.append(float(row[1]))
cell_data = np.array(cell_data)
pos_data = np.array(pos_data)
print("N seconds:", len(cell_data)/200)
print(np.mean(cell_data), np.mean(pos_data))

plt.figure(figsize=(10, 5))

x_values = np.linspace(0, 30, len(cell_data))

y_values = cell_data-np.mean(cell_data)
cell_amp = np.max(cell_data)-np.min(cell_data)
y_values = y_values/cell_amp
plt.plot(x_values, y_values, label="CELL", color="red")

y_values = pos_data-np.mean(pos_data)
pos_amp = np.max(pos_data)-np.min(pos_data)
y_values = y_values/pos_amp
plt.plot(x_values, y_values, label="POS", color="green")

plt.title(f"Data vs time")
plt.xlabel("LABEL")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plot_fft_spectrum(cell_data, fs=200, n_peaks=5)
plot_fft_spectrum(pos_data, fs=200, n_peaks=5)
