import sensor, image, time, math, pyb
from ulab import numpy as np

class MOSSE:
    def __init__(self, lr=0.125, sigma=100, num_pretrain=40):
        self.lr = lr
        self.sigma = sigma
        self.num_pretrain = num_pretrain
        self.Ai = None
        self.Bi = None
        self.pos = None
        self.size = None

    def start(self, init_img: image.Image, bbox):
        """Initialize tracker with first frame and bounding box [x, y, w, h]"""
        init_img = init_img.to_grayscale().to_ndarray(dtype='f')
        x, y, w, h = bbox
        h = self._next_pow2(h)    # pad to power of 2
        w = self._next_pow2(w)    # pad to power of 2
        self.pos = [x, y, w, h]   # resulting bounding box works with fft
        response_map = self._get_gauss_response(init_img, self.pos)
        g = response_map[self.pos[1]:self.pos[1]+self.pos[3], self.pos[0]:self.pos[0]+self.pos[2]]
        fi = init_img[self.pos[1]:self.pos[1]+self.pos[3], self.pos[0]:self.pos[0]+self.pos[2]]
        G = self._fft2(g)
        self.Ai, self.Bi = self._pre_training(fi, G)
        
        return True

    def _get_gauss_response(self, img: np.ndarray, bbox: list):
        """Create Gaussian response map"""
        h, w = img.shape
        y = np.arange(h, dtype=np.float).reshape((h, 1))
        x = np.arange(w, dtype=np.float).reshape((1, w))
        center_x = bbox[0] + 0.5 * bbox[2]
        center_y = bbox[1] + 0.5 * bbox[3]
        dx = x - center_x
        dy = y - center_y
        dist = (dx**2 + dy**2) / (2 * self.sigma)
        response = np.exp(-dist)
        return (response - np.min(response)) / (np.max(response) - np.min(response))

    def _fft2(self, real, imag=None):
        if imag is None:
            imag = np.zeros(real.shape, dtype=np.float)
        rows, cols = real.shape
        for r in range(rows):
            real[r], imag[r] = np.fft.fft(real[r], imag[r])
        real = real.T
        imag = imag.T
        for r in range(cols):  
            real[r], imag[r] = np.fft.fft(real[r], imag[r])
        real = real.T
        imag = imag.T

        return real, imag

    def _pre_training(self, init_frame, G):
        """Pre-train the filter with random warps"""
        h, w = init_frame.shape
        
        # Initialize random number generator
        rng = np.random.Generator(42)  # Fixed seed for reproducibility
        
        # Process and add the original frame
        fi = self._pre_process(init_frame)
        F = self._fft2(fi)
        Fc = self._conj(F)
        # Ai = G * conj(F)
        Ai = self._complex_mul(G, Fc)
        # Bi = F * conj(F)
        Bi = self._complex_mul(F, Fc)
        
        # Add variations of the frame
        for _ in range(self.num_pretrain - 1):
            # Generate random translations using ulab's random
            rand_vals = rng.random(size=(2,))  # Get 2 random floats in [0, 1]
            dx = int((rand_vals[0] * 2 - 1) * (w//8))  # Scale to [-w/8, w/8]
            dy = int((rand_vals[1] * 2 - 1) * (h//8))  # Scale to [-h/8, h/8]
            
            # Apply roll separately for each axis
            warped = fi.copy()
            if dx != 0:
                warped = np.roll(warped, dx, axis=1)
            if dy != 0:
                warped = np.roll(warped, dy, axis=0)
            
            # Process the warped frame
            F_warped = self._fft2(warped)
            Ai_shifted = self._complex_mul(G, self._conj(F_warped))
            Bi_shifted = self._complex_mul(F_warped, self._conj(F_warped))
            Ai = self._complex_add(Ai, Ai_shifted)
            Bi = self._complex_add(Bi, Bi_shifted)
            
        return Ai, Bi

    def _next_pow2(self, n):
        p = 1
        while p < n:
            p <<= 1
        return p

    def _pre_process(self, img):
        """Pre-process image: convert to float32, log transform, normalize, apply window"""
        # Log transform
        # img_data = np.log(img_data + 1.0)
        
        # Normalize
        mean = np.mean(img)
        std = np.std(img)
        img_data = (img - mean) / (std + 1e-5)
        
        # Apply Hanning window
        window = self._hanning_2d(img_data.shape[0], img_data.shape[1])

        return img_data * window

    def _hanning_1d(self, N):
        # ulab-friendly Hanning generator
        n = np.arange(N, dtype=np.float)
        return 0.5 - 0.5 * np.cos(2 * 3.14159 * n / (N - 1))

    def _hanning_2d(self, height, width):
        win_row = self._hanning_1d(height)
        win_col = self._hanning_1d(width)

        # reshape for broadcasting in ulab
        win_row = win_row.reshape((height, 1))
        win_col = win_col.reshape((1, width))

        return win_row * win_col

    def _complex_mul(self, A, B):
        Ar, Ai = A
        Br, Bi = B
        return Ar*Br - Ai*Bi, Ar*Bi + Ai*Br

    def _conj(self, pair):
        real, imag = pair
        return real, -imag

    def _complex_add(self, A, B):
        Ar, Ai = A
        Br, Bi = B
        return Ar+Br, Ai+Bi

