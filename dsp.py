import numpy as np
import matplotlib.pyplot as plt

def blackman_harris_window(win):
    """
    Create a Blackman-Harris Window
    
    Parameters
    ----------
    win: int
        Length of window
    
    Returns
    -------
    ndarray(N): Samples of the window
    """
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    t = np.arange(win)/win
    return a0 - a1*np.cos(2*np.pi*t) + a2*np.cos(4*np.pi*t) - a3*np.cos(6*np.pi*t)

def hann_window(win):
    """
    Create a Hann Window
    
    Parameters
    ----------
    win: int
        Length of window
    
    Returns
    -------
    ndarray(N): Samples of the window
    """
    return 0.5*(1 - np.cos(2*np.pi*np.arange(win)/win))

def stft(x, w, h, win_fn=blackman_harris_window):
    """
    Compute the complex Short-Time Fourier Transform (STFT)
    Parameters
    ----------
    x: ndarray(N)
        Full audio clip of N samples
    w: int
        Window length
    h: int
        Hop length
    win_fn: int -> ndarray(N)
        Window function
    
    Returns
    -------
    ndarray(w, nwindows, dtype=np.complex) STFT
    """
    N = len(x)
    nwin = int(np.ceil((N-w)/h))+1
    # Make a 2D array
    # The rows correspond to frequency bins
    # The columns correspond to windows moved forward in time
    S = np.zeros((w, nwin), dtype=np.complex)
    # Loop through all of the windows, and put the fourier
    # transform amplitudes of each window in its own column
    for j in range(nwin):
        # Pull out the audio in the jth window
        xj = x[h*j:h*j+w]
        # Zeropad if necessary
        if len(xj) < w:
            xj = np.concatenate((xj, np.zeros(w-len(xj))))
        # Apply window function
        xj = win_fn(w)*xj
        # Put the fourier transform into S
        S[:, j] = np.fft.fft(xj)
    return S

def amplitude_to_db(S, amin=1e-10, ref=1):
    """
    Convert an amplitude spectrogram to be expressed in decibels
    
    Parameters
    ----------
    S: ndarray(win, T)
        Amplitude spectrogram
    amin: float
        Minimum accepted value for the spectrogram
    ref: int
        0dB reference amplitude
        
    Returns
    -------
    ndarray(win, T)
        The dB spectrogram
    """
    SLog = 20.0*np.log10(np.maximum(amin, S))
    SLog -= 20.0*np.log10(np.maximum(amin, ref))
    return SLog

def istft(S, w, h):
    """
    Compute the complex inverse Short-Time Fourier Transform (STFT)
    Parameters
    ----------
    S: ndarray(w, nwindows, dtype=np.complex)
        Complex spectrogram
    w: int
        Window length
    h: int
        Hop length
    
    Returns
    -------
    y: ndarray(N)
        Audio samples of the inverted STFT
    """
    N = (S.shape[1]-1)*h + w # Number of samples in result
    y = np.zeros(N)
    for j in range(S.shape[1]):
        xj = np.fft.ifft(S[:, j])
        y[j*h:j*h+w] += np.real(xj)
    y /= (w/h/2)
    return y


def plot_stft(S, sr, hop):
    """
    Plot the spectrogram associated to a short-time
    Fourier Transform, using a log scale

    Parameters
    ----------
    S: ndarray(win, n_frames, dtype=complex):
        Short-time fourier transform
    sr: int
        Sample rate
    hop: int
        Hop length between frames
    """
    win = S.shape[0]
    S = amplitude_to_db(np.abs(S))
    plt.figure()
    plt.imshow(S, extent=(0, S.shape[1]*hop/sr, 0, sr), aspect='auto', cmap='magma')
    plt.xlabel("Time (Sec)")
    plt.ylabel("Frequency (hz)")


def get_spectrogram_image_html(S, hop, sr, fmax=8000, title="", figsize=(6, 3)):
    """
    Create HTML code with base64 binary to display a spectrogram
    
    Parameters
    ----------
    S: ndarray(win, n_frames)
        Spectrogram
    hop: int
        Hop length
    sr: int
        Sample rate
    fmax: int
        Maximum frequency in hz
    title: string
        Title of the plot
    
    Returns
    -------
    string: HTML code with base64 binary blob of matplotlib image
    """
    import matplotlib.pyplot as plt
    import base64
    import io 
    plt.figure(figsize=figsize)
    plt.imshow(amplitude_to_db(S, amin=np.quantile(S, 0.1)), extent=(0, hop*S.shape[1]/sr, 0, sr), aspect='auto', cmap='magma')
    plt.ylim([0, fmax])
    plt.xlabel("Time (Sec)")
    plt.ylabel("Frequency (hz)")
    plt.tight_layout()
    
    # https://stackoverflow.com/questions/38061267/matplotlib-graphic-image-to-base64
    sstream = io.BytesIO()
    plt.savefig(sstream, format='png')
    sstream.seek(0)
    blob = base64.b64encode(sstream.read())
    s = "<img src=\"data:image/png;base64,{}\">".format(blob.decode())
    plt.clf()
    return s

def gl(SAbs, w, h, win_fn, n_iters):
    S = SAbs
    for i in range(n_iters):
        A = stft(istft(S, w, h), w, h, win_fn)
        Phase = np.arctan2(np.imag(A), np.real(A))
        S = SAbs*np.exp(np.complex(0, 1)*Phase)
    x = istft(S, w, h)
    return np.real(x)