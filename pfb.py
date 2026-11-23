### IMPORTS ###

from google.colab import drive, runtime
drive.mount("/content/drive")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2 as cv
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import cupyx.scipy.ndimage as image

from tqdm import tqdm
from os import listdir
from random import seed

seed(0)
np.random.seed(0)
cp.random.seed(0)

### DATA LOADING ###

def readimg(file):
    img = cv.imread(file, cv.IMREAD_GRAYSCALE)
    return cp.asarray(img, dtype=cp.float32)

def readMPI(file, val, occ, occlude):
    with open(file, "rb") as f:
        if np.fromfile(f, np.float32, count=1) != 202021.25:
            raise ValueError(f"Invalid .flo file: {file}")
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        flo = np.fromfile(f, np.float32, count=(2 * w * h)).reshape(h, w, 2)
    flo[cv.imread(val, cv.IMREAD_GRAYSCALE) > 0] = np.nan
    if occlude is None:
        return cp.asarray(flo, dtype=cp.float32)
    occ = cv.imread(occ, cv.IMREAD_GRAYSCALE)
    flo[occ > 0 if occlude else occ == 0] = np.nan
    return cp.asarray(flo, dtype=cp.float32)

def readKITTI(file):
    flo = cv.imread(file, cv.IMREAD_UNCHANGED).astype(np.float32)
    flo[..., 2] = (flo[..., 2] - 2 ** 15) / 64
    flo[..., 1] = (flo[..., 1] - 2 ** 15) / 64
    flo[..., 2][flo[..., 0] == 0] = np.nan
    flo[..., 1][flo[..., 0] == 0] = np.nan
    return cp.asarray(flo[..., [2, 1]], dtype=cp.float32)

def loadpair(dataset, frame=None, sequence=None, render=None, occlude=None):
    base = "/content/training/"
    if dataset == "mpisintel":
        img1 = base + f"{render}/{sequence}/frame_{frame:04}.png"
        img2 = base + f"{render}/{sequence}/frame_{(frame + 1):04}.png"
        flow = base + f"flow/{sequence}/frame_{frame:04}.flo"
        val  = base + f"invalid/{sequence}/frame_{frame:04}.png"
        occ  = base + f"occlusion/{sequence}/frame_{frame:04}.png"
        return readimg(img1), readimg(img2), readMPI(flow, val, occ, occlude)
    if dataset == "kitti2015":
        img1 = base + f"image_2/{frame:06}_10.png"
        img2 = base + f"image_2/{frame:06}_11.png"
        flow = "flow_noc" if occlude is False else "flow_occ"
        flow = base + f"{flow}/{frame:06}_10.png"
        return readimg(img1), readimg(img2), readKITTI(flow)

### VISUALIZATION ###

def visimg(img1, img2, figsize=(10, 4)):
    fig, axs = plt.subplots(ncols=2, figsize=figsize)
    axs[0].imshow(img1, cmap="grey")
    axs[1].imshow(img2, cmap="grey")
    axs[0].axis("off")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()

def colorwheel():
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    wheel = cp.zeros((ncols, 3), dtype=cp.float32)
    col = 0

    # RY (Red -> Yellow)
    wheel[col:col+RY, 0] = 255
    wheel[col:col+RY, 1] = cp.floor(255 * cp.arange(RY, dtype=cp.float32) / RY)
    col += RY

    # YG (Yellow -> Green)
    wheel[col:col+YG, 0] = 255 - cp.floor(255 * cp.arange(YG, dtype=cp.float32) / YG)
    wheel[col:col+YG, 1] = 255
    col += YG

    # GC (Green -> Cyan)
    wheel[col:col+GC, 1] = 255
    wheel[col:col+GC, 2] = cp.floor(255 * cp.arange(GC, dtype=cp.float32) / GC)
    col += GC

    # CB (Cyan -> Blue)
    wheel[col:col+CB, 1] = 255 - cp.floor(255 * cp.arange(CB, dtype=cp.float32) / CB)
    wheel[col:col+CB, 2] = 255
    col += CB

    # BM (Blue -> Magenta)
    wheel[col:col+BM, 2] = 255
    wheel[col:col+BM, 0] = cp.floor(255 * cp.arange(BM, dtype=cp.float32) / BM)
    col += BM

    # MR (Magenta -> Red)
    wheel[col:col+MR, 2] = 255 - cp.floor(255 * cp.arange(MR, dtype=cp.float32) / MR)
    wheel[col:col+MR, 0] = 255
    return wheel

def computeColor(u, v):
    nan_idx = cp.isnan(u) | cp.isnan(v) | (cp.abs(u) > 1e9) | (cp.abs(v) > 1e9)
    u = cp.where(nan_idx, 0, u)
    v = cp.where(nan_idx, 0, v)

    wheel = colorwheel()
    ncols = wheel.shape[0]
    rad = cp.sqrt(u**2 + v**2)
    a = cp.atan2(-v, -u) / cp.pi

    fk = (a + 1.0) / 2.0 * (ncols - 1.0)
    k0 = cp.floor(fk).astype(int)
    f = fk - k0
    k1 = (k0 + 1) % ncols
    k0 = k0 % ncols

    img = cp.zeros((u.shape[0], u.shape[1], 3), dtype=cp.float32)

    for i in range(3):
        tmp = wheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1.0 - f) * col0 + f * col1

        idx = rad <= 1.0
        col = cp.where(idx, 1.0 - rad * (1.0 - col), col * 0.75)
        img[..., i] = cp.where(nan_idx, 0.0, col)
    return img

def flow2color(u, v, max_flow=None):
    invalid = cp.isnan(u) | cp.isnan(v) | (cp.abs(u) > 1e9) | (cp.abs(v) > 1e9)
    u_clean = cp.where(invalid, 0, u)
    v_clean = cp.where(invalid, 0, v)

    rad = cp.sqrt(u_clean ** 2 + v_clean ** 2)
    maxrad = float(cp.max(rad))
    if maxrad < 1e-5:
        maxrad = 1.0
    if max_flow is not None and max_flow > 0:
        maxrad = float(max_flow)

    u_norm = u_clean / (maxrad + 1e-5)
    v_norm = v_clean / (maxrad + 1e-5)
    img = computeColor(u_norm, v_norm)

    for i in range(3):
        img[..., i] = cp.where(invalid, 0.0, img[..., i])
    return img

def visflo(u, v, c, gtmax=None, log=False, scale=None):
    img_flo = flow2color(u, v)
    eigenvalues, eigenvectors = cp.linalg.eigh(c)
    lam_max = cp.maximum(0.0, eigenvalues[..., 1])
    vec_max = eigenvectors[..., :, 1]

    dot = vec_max[..., 0] * u + vec_max[..., 1] * v
    vec_max = cp.where(dot[..., None] < 0, -vec_max, vec_max)
    mag_cov = cp.log1p(scale * lam_max) if log else lam_max

    u_cov = mag_cov * vec_max[..., 0]
    v_cov = mag_cov * vec_max[..., 1]
    img_cov = flow2color(u_cov, v_cov, gtmax)
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))

    axs[0].imshow(img_flo.get())
    axs[1].imshow(img_cov.get())
    axs[0].axis("off")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()

### PFB AUXILIARY FUNCTIONS ###

def gaussianPyramid(img, lvls):
    pyr = [img.astype(cp.float32)]
    ker = cp.array(
       [[1,  4,  6,  4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4,  6,  4, 1]],
        dtype=cp.float32
    ) / 256.0
    for _ in range(lvls - 1):
        blurred = image.convolve(pyr[-1], ker, mode="mirror")
        pyr.append(blurred[::2, ::2])
    return pyr[::-1]

def gaussianWeights(ksize, sigma):
    zz = cp.arange(-(ksize // 2), ksize // 2 + 1, dtype=cp.float32)
    xx, yy = cp.meshgrid(zz, zz)
    return cp.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

def gramSchmidt(W, B, normal=True):
    Q = cp.zeros_like(B)
    for j in range(B.shape[1]):
        v = B[:, j].copy()
        for i in range(j):
            qi = Q[:, i]
            num = qi @ W @ v
            den = qi @ W @ qi
            if den > 1e-8:
                v -= (num / den) * qi
        Q[:, j] = v
    if not normal:
        return Q
    for j in range(Q.shape[1]):
        norm = cp.sqrt(Q[:, j] @ W @ Q[:, j])
        if norm > 1e-8:
            Q[:, j] /= norm
    return Q

def generateFarnebackFilters(ksize, sigma, ortho=False):
    zz = cp.arange(-(ksize // 2), ksize // 2 + 1, dtype=cp.float32)
    xx, yy = cp.meshgrid(zz, zz)
    W = cp.diag(cp.exp(-(xx * xx + yy * yy) / (2 * sigma ** 2)).ravel())
    B = [cp.ones_like(xx), xx, yy, xx * xx, yy * yy, xx * yy]
    B = cp.stack(B, axis=-1).reshape(-1, 6)
    B = gramSchmidt(W, B) if ortho else B
    BtW = B.T @ W
    BtWB = BtW @ B
    M = cp.linalg.inv(BtWB) @ BtW
    return M.reshape(6, ksize, ksize)

### PFB-INTENSITY ###

def kernelI():
    return cp.ElementwiseKernel(
        in_params="raw float32 Y0, raw float32 y0, raw float32 u, raw float32 v, "
                  "raw float32 p0x, raw float32 p0y, raw float32 C1, raw float32 C2, "
                  "raw float32 Bx1, raw float32 By1, raw float32 Bx2, raw float32 By2, "
                  "raw float32 W, float32 gamma, int32 ksize, int32 ph, int32 pw",
        out_params="raw float32 u_new, raw float32 v_new, raw float32 c_new",
        operation=r"""
        int pt_idx = i;
        int k = ksize / 2;

        float px = p0x[pt_idx];
        float py = p0y[pt_idx];
        float u_val = u[pt_idx];
        float v_val = v[pt_idx];

        int rpy = max(0, min((int)round(py), ph - 2*k - 1));
        int rpx = max(0, min((int)round(px), pw - 2*k - 1));
        int m = rpy + k;
        int n = rpx + k;

        float Y00 = Y0[4 * pt_idx + 0]; float Y01 = Y0[4 * pt_idx + 1];
        float Y10 = Y0[4 * pt_idx + 2]; float Y11 = Y0[4 * pt_idx + 3];
        float y00 = y0[2 * pt_idx + 0]; float y01 = y0[2 * pt_idx + 1];

        for (int di = -k; di <= k; di++) {
            for (int dj = -k; dj <= k; dj++) {
                int y1 = m + di;
                int x1 = n + dj;
                float x2 = x1 + u_val;
                float y2 = y1 + v_val;

                int x0 = (int)floor(x2); int y0 = (int)floor(y2);
                float fx = x2 - x0; float fy = y2 - y0;
                x0 = max(0, min(x0, pw - 1)); y0 = max(0, min(y0, ph - 1));
                int x1b = min(x0 + 1, pw - 1); int y1b = min(y0 + 1, ph - 1);

                #define BILINEAR(Q) ((1 - fx) * (1 - fy) * Q[y0 * pw + x0] + \
                                     fx * (1 - fy) * Q[y0 * pw + x1b] + \
                                     (1 - fx) * fy * Q[y1b * pw + x0] + \
                                     fx * fy * Q[y1b * pw + x1b])

                float c1  = C1[y1 * pw + x1];
                float bx1 = Bx1[y1 * pw + x1];
                float by1 = By1[y1 * pw + x1];

                float c2  = BILINEAR(C2);
                float bx2 = BILINEAR(Bx2);
                float by2 = BILINEAR(By2);
                #undef BILINEAR

                float it = c1 - c2;
                float ix = 0.5f * (bx1 + bx2);
                float iy = 0.5f * (by1 + by2);
                float rv = max(1e-6f, powf(fabs(it), gamma));
                float wv = W[(di + k) * ksize + (dj + k)];
                float inv_rv = wv / rv;
                float pseudo = it + (ix * u_val + iy * v_val);

                Y00 += ix * ix * inv_rv;
                Y01 += ix * iy * inv_rv;
                Y10 += iy * ix * inv_rv;
                Y11 += iy * iy * inv_rv;
                y00 += ix * pseudo * inv_rv;
                y01 += iy * pseudo * inv_rv;
            }
        }

        float det = Y00 * Y11 - Y01 * Y10;
        c_new[4 * pt_idx + 0] =  Y11 / det;
        c_new[4 * pt_idx + 1] = -Y01 / det;
        c_new[4 * pt_idx + 2] = -Y10 / det;
        c_new[4 * pt_idx + 3] =  Y00 / det;
        u_new[pt_idx] = c_new[4 * pt_idx + 0] * y00 + c_new[4 * pt_idx + 1] * y01;
        v_new[pt_idx] = c_new[4 * pt_idx + 2] * y00 + c_new[4 * pt_idx + 3] * y01;
        """
    )

def PFBI(img1, img2, p0, lvls, ksize, eps, gamma, sigma, fksize, fsigma, niter, tol):
    N = p0.shape[0]
    pyr1 = gaussianPyramid(img1, lvls)
    pyr2 = gaussianPyramid(img2, lvls)

    F, Fx, Fy, _, _, _ = generateFarnebackFilters(fksize, fsigma)
    W = gaussianWeights(ksize, sigma).astype(cp.float32)
    update_kernel = kernelI()

    p0x = p0[:, 0].astype(cp.float32)
    p0y = p0[:, 1].astype(cp.float32)
    k = ksize // 2

    for level in range(lvls):
        if not level:
            u = cp.zeros(N, dtype=cp.float32)
            v = cp.zeros(N, dtype=cp.float32)
            c = cp.zeros((N, 2, 2), dtype=cp.float32)
            c[:, 0, 0] = c[:, 1, 1] = 1e6
        else:
            u *= 2.0
            v *= 2.0
            c *= 4.0
            c[:, 0, 0] += eps
            c[:, 1, 1] += eps
            c[:, 1, 0] = c[:, 0, 1]

        scale = 2 ** (lvls - 1 - level)
        p_lvl_x = p0x / scale
        p_lvl_y = p0y / scale

        pad1 = cp.pad(pyr1[level], k, mode="reflect")
        pad2 = cp.pad(pyr2[level], k, mode="reflect")
        ph, pw = pad1.shape

        C1   = image.correlate(pad1, F,   mode="mirror")
        Bx1  = image.correlate(pad1, Fx,  mode="mirror")
        By1  = image.correlate(pad1, Fy,  mode="mirror")
        C2   = image.correlate(pad2, F,   mode="mirror")
        Bx2  = image.correlate(pad2, Fx,  mode="mirror")
        By2  = image.correlate(pad2, Fy,  mode="mirror")

        inv_c = cp.empty_like(c)
        det_c = c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1] * c[:, 1, 0]
        det_c[det_c < 1e-6] = 1e-6

        inv_c[:, 0, 0] =  c[:, 1, 1] / det_c
        inv_c[:, 1, 1] =  c[:, 0, 0] / det_c
        inv_c[:, 0, 1] = -c[:, 0, 1] / det_c
        inv_c[:, 1, 0] = -c[:, 1, 0] / det_c

        Y_prior = inv_c.reshape(N, 4).astype(cp.float32)
        y_prior = cp.einsum(
            "...ij,...j->...i", inv_c, cp.stack((u, v), axis=-1)
        ).astype(cp.float32)

        u0, v0 = u.copy(), v.copy()
        c_new = cp.zeros_like(c)

        for _ in range(niter):
            u_new = cp.zeros_like(u)
            v_new = cp.zeros_like(v)

            update_kernel(
                Y_prior, y_prior, u0, v0,
                p_lvl_x, p_lvl_y, C1, C2,
                Bx1, By1, Bx2, By2,
                W, gamma, ksize, ph, pw,
                u_new, v_new, c_new, size=N
            )
            du = u_new - u0
            dv = v_new - v0
            u0, v0 = u_new, v_new
            if cp.sqrt(cp.mean(du ** 2 + dv ** 2)) < tol:
                break

        update_kernel(
            Y_prior, y_prior, u0, v0,
            p_lvl_x, p_lvl_y, C1, C2,
            Bx1, By1, Bx2, By2,
            W, gamma, ksize, ph, pw,
            u_new, v_new, c_new, size=N
        )
        u, v, c = u0, v0, c_new
    return u, v, c

### PFB-GRADIENT ###

def kernelG():
    return cp.ElementwiseKernel(
        in_params="raw float32 Y0, raw float32 y0, raw float32 u, raw float32 v, "
                  "raw float32 p0x, raw float32 p0y, "
                  "raw float32 Bx1, raw float32 By1, raw float32 Axx1, raw float32 Ayy1, raw float32 Axy1, "
                  "raw float32 Bx2, raw float32 By2, raw float32 Axx2, raw float32 Ayy2, raw float32 Axy2, "
                  "raw float32 W, float32 gamma, int32 ksize, int32 ph, int32 pw",
        out_params="raw float32 u_new, raw float32 v_new, raw float32 c_new",
        operation=r"""
        int pt_idx = i;
        int k = ksize / 2;

        float px = p0x[pt_idx];
        float py = p0y[pt_idx];
        float u_val = u[pt_idx];
        float v_val = v[pt_idx];

        int rpy = max(0, min((int)round(py), ph - 2*k - 1));
        int rpx = max(0, min((int)round(px), pw - 2*k - 1));
        int m = rpy + k;
        int n = rpx + k;

        float Y00 = Y0[4 * pt_idx + 0]; float Y01 = Y0[4 * pt_idx + 1];
        float Y10 = Y0[4 * pt_idx + 2]; float Y11 = Y0[4 * pt_idx + 3];
        float y00 = y0[2 * pt_idx + 0]; float y01 = y0[2 * pt_idx + 1];

        for (int di = -k; di <= k; di++) {
            for (int dj = -k; dj <= k; dj++) {
                int y1 = m + di;
                int x1 = n + dj;
                float x2 = x1 + u_val;
                float y2 = y1 + v_val;

                int x0 = (int)floor(x2); int y0 = (int)floor(y2);
                float fx = x2 - x0; float fy = y2 - y0;
                x0 = max(0, min(x0, pw - 1)); y0 = max(0, min(y0, ph - 1));
                int x1b = min(x0 + 1, pw - 1); int y1b = min(y0 + 1, ph - 1);

                #define BILINEAR(Q) ((1 - fx) * (1 - fy) * Q[y0 * pw + x0] + \
                                     fx * (1 - fy) * Q[y0 * pw + x1b] + \
                                     (1 - fx) * fy * Q[y1b * pw + x0] + \
                                     fx * fy * Q[y1b * pw + x1b])

                float bx1  = Bx1[y1 * pw + x1];
                float by1  = By1[y1 * pw + x1];
                float axx1 = Axx1[y1 * pw + x1];
                float ayy1 = Ayy1[y1 * pw + x1];
                float axy1 = Axy1[y1 * pw + x1];

                float bx2  = BILINEAR(Bx2);
                float by2  = BILINEAR(By2);
                float axx2 = BILINEAR(Axx2);
                float ayy2 = BILINEAR(Ayy2);
                float axy2 = BILINEAR(Axy2);
                #undef BILINEAR

                float hxx = /* 0.5f */ (axx1 + axx2);
                float hyy = /* 0.5f */ (ayy1 + ayy2);
                float hxy = /* 0.5f */ (axy1 + axy2);
                float itx = bx1 - bx2;
                float ity = by1 - by2;

                float rvx = max(1e-6f, powf(fabs(itx), gamma));
                float rvy = max(1e-6f, powf(fabs(ity), gamma));
                float wv = W[(di + k) * ksize + (dj + k)];
                float inv_rvx = wv / rvx;
                float inv_rvy = wv / rvy;
                float pseudox = itx + (hxx * u_val + hxy * v_val);
                float pseudoy = ity + (hxy * u_val + hyy * v_val);

                Y00 += (hxx * hxx * inv_rvx) + (hxy * hxy * inv_rvy);
                Y01 += (hxx * hxy * inv_rvx) + (hxy * hyy * inv_rvy);
                Y10 += (hxx * hxy * inv_rvx) + (hxy * hyy * inv_rvy);
                Y11 += (hxy * hxy * inv_rvx) + (hyy * hyy * inv_rvy);
                y00 += (hxx * pseudox * inv_rvx) + (hxy * pseudoy * inv_rvy);
                y01 += (hxy * pseudox * inv_rvx) + (hyy * pseudoy * inv_rvy);
            }
        }

        float det = Y00 * Y11 - Y01 * Y10;
        c_new[4 * pt_idx + 0] =  Y11 / det;
        c_new[4 * pt_idx + 1] = -Y01 / det;
        c_new[4 * pt_idx + 2] = -Y10 / det;
        c_new[4 * pt_idx + 3] =  Y00 / det;
        u_new[pt_idx] = c_new[4 * pt_idx + 0] * y00 + c_new[4 * pt_idx + 1] * y01;
        v_new[pt_idx] = c_new[4 * pt_idx + 2] * y00 + c_new[4 * pt_idx + 3] * y01;
        """
    )

def PFBG(img1, img2, p0, lvls, ksize, eps, gamma, sigma, fksize, fsigma, niter, tol):
    N = p0.shape[0]
    pyr1 = gaussianPyramid(img1, lvls)
    pyr2 = gaussianPyramid(img2, lvls)

    _, Fx, Fy, Fxx, Fyy, Fxy = generateFarnebackFilters(fksize, fsigma)
    W = gaussianWeights(ksize, sigma).astype(cp.float32)
    update_kernel = kernelG()

    p0x = p0[:, 0].astype(cp.float32)
    p0y = p0[:, 1].astype(cp.float32)
    k = ksize // 2

    for level in range(lvls):
        if not level:
            u = cp.zeros(N, dtype=cp.float32)
            v = cp.zeros(N, dtype=cp.float32)
            c = cp.zeros((N, 2, 2), dtype=cp.float32)
            c[:, 0, 0] = c[:, 1, 1] = 1e6
        else:
            u *= 2.0
            v *= 2.0
            c *= 4.0
            c[:, 0, 0] += eps
            c[:, 1, 1] += eps
            c[:, 1, 0] = c[:, 0, 1]

        scale = 2 ** (lvls - 1 - level)
        p_lvl_x = p0x / scale
        p_lvl_y = p0y / scale

        pad1 = cp.pad(pyr1[level], k, mode="reflect")
        pad2 = cp.pad(pyr2[level], k, mode="reflect")
        ph, pw = pad1.shape

        Bx1  = image.correlate(pad1, Fx,  mode="mirror")
        By1  = image.correlate(pad1, Fy,  mode="mirror")
        Axx1 = image.correlate(pad1, Fxx, mode="mirror")
        Ayy1 = image.correlate(pad1, Fyy, mode="mirror")
        Axy1 = image.correlate(pad1, Fxy, mode="mirror")

        Bx2  = image.correlate(pad2, Fx,  mode="mirror")
        By2  = image.correlate(pad2, Fy,  mode="mirror")
        Axx2 = image.correlate(pad2, Fxx, mode="mirror")
        Ayy2 = image.correlate(pad2, Fyy, mode="mirror")
        Axy2 = image.correlate(pad2, Fxy, mode="mirror")

        inv_c = cp.empty_like(c)
        det_c = c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1] * c[:, 1, 0]
        det_c[det_c < 1e-6] = 1e-6

        inv_c[:, 0, 0] =  c[:, 1, 1] / det_c
        inv_c[:, 1, 1] =  c[:, 0, 0] / det_c
        inv_c[:, 0, 1] = -c[:, 0, 1] / det_c
        inv_c[:, 1, 0] = -c[:, 1, 0] / det_c

        Y_prior = inv_c.reshape(N, 4).astype(cp.float32)
        y_prior = cp.einsum(
            "...ij,...j->...i", inv_c, cp.stack((u, v), axis=-1)
        ).astype(cp.float32)

        u0, v0 = u.copy(), v.copy()
        c_new = cp.zeros_like(c)

        for _ in range(niter):
            u_new = cp.zeros_like(u)
            v_new = cp.zeros_like(v)

            update_kernel(
                Y_prior, y_prior, u0, v0,
                p_lvl_x, p_lvl_y,
                Bx1, By1, Axx1, Ayy1, Axy1,
                Bx2, By2, Axx2, Ayy2, Axy2,
                W, gamma, ksize, ph, pw,
                u_new, v_new, c_new, size=N
            )
            du = u_new - u0
            dv = v_new - v0
            u0, v0 = u_new, v_new
            if cp.sqrt(cp.mean(du ** 2 + dv ** 2)) < tol:
                break

        update_kernel(
            Y_prior, y_prior, u0, v0,
            p_lvl_x, p_lvl_y,
            Bx1, By1, Axx1, Ayy1, Axy1,
            Bx2, By2, Axx2, Ayy2, Axy2,
            W, gamma, ksize, ph, pw,
            u_new, v_new, c_new, size=N
        )
        u, v, c = u0, v0, c_new
    return u, v, c

### PFB-COMBINED ###

def kernelC():
    return cp.ElementwiseKernel(
        in_params="raw float32 Y0, raw float32 y0, raw float32 u, raw float32 v, "
                  "raw float32 p0x, raw float32 p0y, raw float32 C1, raw float32 C2, "
                  "raw float32 Bx1, raw float32 By1, raw float32 Axx1, raw float32 Ayy1, raw float32 Axy1, "
                  "raw float32 Bx2, raw float32 By2, raw float32 Axx2, raw float32 Ayy2, raw float32 Axy2, "
                  "raw float32 W, float32 gamma, int32 ksize, int32 ph, int32 pw",
        out_params="raw float32 u_new, raw float32 v_new, raw float32 c_new",
        operation=r"""
        int pt_idx = i;
        int k = ksize / 2;

        float px = p0x[pt_idx];
        float py = p0y[pt_idx];
        float u_val = u[pt_idx];
        float v_val = v[pt_idx];

        int rpy = max(0, min((int)round(py), ph - 2*k - 1));
        int rpx = max(0, min((int)round(px), pw - 2*k - 1));
        int m = rpy + k;
        int n = rpx + k;

        float Y00 = Y0[4 * pt_idx + 0]; float Y01 = Y0[4 * pt_idx + 1];
        float Y10 = Y0[4 * pt_idx + 2]; float Y11 = Y0[4 * pt_idx + 3];
        float y00 = y0[2 * pt_idx + 0]; float y01 = y0[2 * pt_idx + 1];

        for (int di = -k; di <= k; di++) {
            for (int dj = -k; dj <= k; dj++) {
                int y1 = m + di;
                int x1 = n + dj;
                float x2 = x1 + u_val;
                float y2 = y1 + v_val;

                int x0 = (int)floor(x2); int y0 = (int)floor(y2);
                float fx = x2 - x0; float fy = y2 - y0;
                x0 = max(0, min(x0, pw - 1)); y0 = max(0, min(y0, ph - 1));
                int x1b = min(x0 + 1, pw - 1); int y1b = min(y0 + 1, ph - 1);

                #define BILINEAR(Q) ((1 - fx) * (1 - fy) * Q[y0 * pw + x0] + \
                                     fx * (1 - fy) * Q[y0 * pw + x1b] + \
                                     (1 - fx) * fy * Q[y1b * pw + x0] + \
                                     fx * fy * Q[y1b * pw + x1b])

                float c1   = C1[y1 * pw + x1];
                float bx1  = Bx1[y1 * pw + x1];
                float by1  = By1[y1 * pw + x1];
                float axx1 = Axx1[y1 * pw + x1];
                float ayy1 = Ayy1[y1 * pw + x1];
                float axy1 = Axy1[y1 * pw + x1];

                float c2   = BILINEAR(C2);
                float bx2  = BILINEAR(Bx2);
                float by2  = BILINEAR(By2);
                float axx2 = BILINEAR(Axx2);
                float ayy2 = BILINEAR(Ayy2);
                float axy2 = BILINEAR(Axy2);
                #undef BILINEAR

                float it = c1 - c2;
                float ix = 0.5f * (bx1 + bx2);
                float iy = 0.5f * (by1 + by2);
                float rv = max(1e-6f, powf(fabs(it), gamma));
                float wv = W[(di + k) * ksize + (dj + k)];
                float inv_rv = wv / rv;
                float pseudo = it + (ix * u_val + iy * v_val);

                Y00 += ix * ix * inv_rv;
                Y01 += ix * iy * inv_rv;
                Y10 += iy * ix * inv_rv;
                Y11 += iy * iy * inv_rv;
                y00 += ix * pseudo * inv_rv;
                y01 += iy * pseudo * inv_rv;

                float hxx = /* 0.5f */ (axx1 + axx2);
                float hyy = /* 0.5f */ (ayy1 + ayy2);
                float hxy = /* 0.5f */ (axy1 + axy2);
                float itx = bx1 - bx2;
                float ity = by1 - by2;

                float rvx = max(1e-6f, powf(fabs(itx), gamma));
                float rvy = max(1e-6f, powf(fabs(ity), gamma));
                float inv_rvx = wv / rvx;
                float inv_rvy = wv / rvy;
                float pseudox = itx + (hxx * u_val + hxy * v_val);
                float pseudoy = ity + (hxy * u_val + hyy * v_val);

                Y00 += (hxx * hxx * inv_rvx) + (hxy * hxy * inv_rvy);
                Y01 += (hxx * hxy * inv_rvx) + (hxy * hyy * inv_rvy);
                Y10 += (hxx * hxy * inv_rvx) + (hxy * hyy * inv_rvy);
                Y11 += (hxy * hxy * inv_rvx) + (hyy * hyy * inv_rvy);
                y00 += (hxx * pseudox * inv_rvx) + (hxy * pseudoy * inv_rvy);
                y01 += (hxy * pseudox * inv_rvx) + (hyy * pseudoy * inv_rvy);
            }
        }

        float det = Y00 * Y11 - Y01 * Y10;
        c_new[4 * pt_idx + 0] =  Y11 / det;
        c_new[4 * pt_idx + 1] = -Y01 / det;
        c_new[4 * pt_idx + 2] = -Y10 / det;
        c_new[4 * pt_idx + 3] =  Y00 / det;
        u_new[pt_idx] = c_new[4 * pt_idx + 0] * y00 + c_new[4 * pt_idx + 1] * y01;
        v_new[pt_idx] = c_new[4 * pt_idx + 2] * y00 + c_new[4 * pt_idx + 3] * y01;
        """
    )

def PFBC(img1, img2, p0, lvls, ksize, eps, gamma, sigma, fksize, fsigma, niter, tol):
    N = p0.shape[0]
    pyr1 = gaussianPyramid(img1, lvls)
    pyr2 = gaussianPyramid(img2, lvls)

    F, Fx, Fy, Fxx, Fyy, Fxy = generateFarnebackFilters(fksize, fsigma)
    W = gaussianWeights(ksize, sigma).astype(cp.float32)
    update_kernel = kernelC()

    p0x = p0[:, 0].astype(cp.float32)
    p0y = p0[:, 1].astype(cp.float32)
    k = ksize // 2

    for level in range(lvls):
        if not level:
            u = cp.zeros(N, dtype=cp.float32)
            v = cp.zeros(N, dtype=cp.float32)
            c = cp.zeros((N, 2, 2), dtype=cp.float32)
            c[:, 0, 0] = c[:, 1, 1] = 1e6
        else:
            u *= 2.0
            v *= 2.0
            c *= 4.0
            c[:, 0, 0] += eps
            c[:, 1, 1] += eps
            c[:, 1, 0] = c[:, 0, 1]

        scale = 2 ** (lvls - 1 - level)
        p_lvl_x = p0x / scale
        p_lvl_y = p0y / scale

        pad1 = cp.pad(pyr1[level], k, mode="reflect")
        pad2 = cp.pad(pyr2[level], k, mode="reflect")
        ph, pw = pad1.shape

        C1   = image.correlate(pad1, F,   mode="mirror")
        Bx1  = image.correlate(pad1, Fx,  mode="mirror")
        By1  = image.correlate(pad1, Fy,  mode="mirror")
        Axx1 = image.correlate(pad1, Fxx, mode="mirror")
        Ayy1 = image.correlate(pad1, Fyy, mode="mirror")
        Axy1 = image.correlate(pad1, Fxy, mode="mirror")

        C2   = image.correlate(pad2, F,   mode="mirror")
        Bx2  = image.correlate(pad2, Fx,  mode="mirror")
        By2  = image.correlate(pad2, Fy,  mode="mirror")
        Axx2 = image.correlate(pad2, Fxx, mode="mirror")
        Ayy2 = image.correlate(pad2, Fyy, mode="mirror")
        Axy2 = image.correlate(pad2, Fxy, mode="mirror")

        inv_c = cp.empty_like(c)
        det_c = c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1] * c[:, 1, 0]
        det_c[det_c < 1e-6] = 1e-6

        inv_c[:, 0, 0] =  c[:, 1, 1] / det_c
        inv_c[:, 1, 1] =  c[:, 0, 0] / det_c
        inv_c[:, 0, 1] = -c[:, 0, 1] / det_c
        inv_c[:, 1, 0] = -c[:, 1, 0] / det_c

        Y_prior = inv_c.reshape(N, 4).astype(cp.float32)
        y_prior = cp.einsum(
            "...ij,...j->...i", inv_c, cp.stack((u, v), axis=-1)
        ).astype(cp.float32)

        u0, v0 = u.copy(), v.copy()
        c_new = cp.zeros_like(c)

        for _ in range(niter):
            u_new = cp.zeros_like(u)
            v_new = cp.zeros_like(v)

            update_kernel(
                Y_prior, y_prior, u0, v0,
                p_lvl_x, p_lvl_y, C1, C2,
                Bx1, By1, Axx1, Ayy1, Axy1,
                Bx2, By2, Axx2, Ayy2, Axy2,
                W, gamma, ksize, ph, pw,
                u_new, v_new, c_new, size=N
            )
            du = u_new - u0
            dv = v_new - v0
            u0, v0 = u_new, v_new
            if cp.sqrt(cp.mean(du ** 2 + dv ** 2)) < tol:
                break

        update_kernel(
            Y_prior, y_prior, u0, v0,
            p_lvl_x, p_lvl_y, C1, C2,
            Bx1, By1, Axx1, Ayy1, Axy1,
            Bx2, By2, Axx2, Ayy2, Axy2,
            W, gamma, ksize, ph, pw,
            u_new, v_new, c_new, size=N
        )
        u, v, c = u0, v0, c_new
    return u, v, c

### FLOW ESTIMATE EVALUATION ###

def computeEPE(true, pred=None, p0=None, p1=None):
    xp = cp.get_array_module(true)

    if p0 is None:
        ee = xp.sqrt(xp.sum((true - pred) ** 2, axis=-1))
        valid = ~xp.any(xp.isnan(true), axis=-1)
    else:
        h, w = true.shape[:2]
        predf = pred if p1 is None else p1 - p0
        x = xp.clip(p0[:, 0].astype(xp.int32), 0, w - 1)
        y = xp.clip(p0[:, 1].astype(xp.int32), 0, h - 1)
        truef = true[y, x]
        ee = xp.sqrt(xp.sum((truef - predf) ** 2, axis=-1))
        valid = ~xp.any(xp.isnan(truef), axis=-1)

    return xp.mean(ee[valid]), valid.sum()

def computeF1(true, pred=None, p0=None, p1=None):
    xp = cp.get_array_module(true)

    if p0 is None:
        ee = xp.sqrt(xp.sum((true - pred) ** 2, axis=-1))
        gtmag = xp.sqrt(xp.sum(true ** 2, axis=-1))
        valid = ~xp.any(xp.isnan(true), axis=-1)
    else:
        h, w = true.shape[:2]
        predf = pred if p1 is None else p1 - p0
        x = xp.clip(p0[:, 0].astype(xp.int32), 0, w - 1)
        y = xp.clip(p0[:, 1].astype(xp.int32), 0, h - 1)
        truef = true[y, x]
        ee = xp.sqrt(xp.sum((truef - predf) ** 2, axis=-1))
        gtmag = xp.sqrt(xp.sum(truef ** 2, axis=-1))
        valid = ~xp.any(xp.isnan(truef), axis=-1)

    return xp.mean((ee[valid] >= 3) & (ee[valid] >= 0.05 * gtmag[valid]))

def run(func, imgs, *args, macro=1, **kwargs):
    epe, f1, nval = [], [], []

    for img1, img2, imgf in tqdm(imgs):
        h, w = img1.shape
        y, x = cp.mgrid[0:h, 0:w].astype(cp.float32)
        pts = cp.stack((x, y), axis=-1).reshape(-1, 2)
        u, v, _ = func(img1, img2, pts, *args, **kwargs)
        ee, nv = computeEPE(imgf, pred := cp.vstack((u, v)).T, pts)
        f1.append(computeF1(imgf, pred, pts))
        epe.append(ee)
        nval.append(nv)

    epe, f1, nval = cp.array(epe), cp.array(f1), cp.array(nval)
    epe = cp.sum(epe * nval) / cp.sum(nval) if macro else cp.mean(epe)
    f1 = cp.sum(f1 * nval) / cp.sum(nval)
    return epe, f1

### UNCERTAINTY ESTIMATE EVALUATION ###

def computeAUC(total, arr, n, chunk):
    acc, runsum = 0, 0
    for s in range(0, n - 1, chunk):
        e = min(s + chunk, n - 1)
        prefix = cp.cumsum(arr[s:e])
        cumval = runsum + prefix
        i = cp.arange(len(prefix), dtype=cp.float64)
        denom = n - (s + i + 1)
        avg = (total - cumval) / denom
        if s == 0:
            avg[0] *= 0.5
        if e == n - 1:
            avg[-1] *= 0.5
        acc += cp.sum(avg)
        runsum += cp.sum(arr[s:e])
    return acc / n

def run(func, imgs, *args, macro=1, chunk=10_000_000, **kwargs):
    errs, f1, epe, nval = [], [], [], []
    uncs0, uncs1, uncs2 = [], [], []

    for img1, img2, imgf in tqdm(imgs):
        h, w = img1.shape
        y, x = cp.mgrid[0:h, 0:w].astype(cp.float32)
        pts = cp.stack((x, y), axis=-1).reshape(-1, 2)
        u, v, c = func(img1, img2, pts, *args, **kwargs)

        x = cp.clip(pts[:, 0].astype(cp.int32), 0, w - 1)
        y = cp.clip(pts[:, 1].astype(cp.int32), 0, h - 1)
        valid = ~cp.any(cp.isnan(truef := imgf[y, x]), axis=-1)
        ee = cp.sqrt(cp.sum((truef - cp.vstack((u, v)).T) ** 2, axis=-1))

        errs.append(err := ee[valid])
        gtmag = cp.sqrt(cp.sum(truef ** 2, axis=-1))
        f1.append(cp.mean((err >= 3) & (err >= 0.05 * gtmag[valid])))
        epe.append(cp.mean(err))
        nval.append(valid.sum())
        unc0 = cp.linalg.eigvalsh(c).sum(axis=-1)

        sobel_x = cp.array([[-1,  0,  1], [-2, 0, 2], [-1,  0, 1]], dtype=cp.float32)
        sobel_y = cp.array([[-1, -2, -1], [ 0, 0, 0], [ 1,  2, 1]], dtype=cp.float32)
        Ix = image.correlate(img1, sobel_x, mode="mirror")
        Iy = image.correlate(img1, sobel_y, mode="mirror")
        Sxx, Syy, Sxy = Ix * Ix, Iy * Iy, Ix * Iy
        box = cp.ones((k := kwargs["ksize"], k), dtype=cp.float32) / (k * k)
        Sxx = image.correlate(Sxx, box, mode="mirror")
        Syy = image.correlate(Syy, box, mode="mirror")
        Sxy = image.correlate(Sxy, box, mode="mirror")
        trace, det = Sxx + Syy, Sxx * Syy - Sxy * Sxy
        disc = cp.sqrt(cp.maximum(trace * trace - 4.0 * det, 0.0))
        unc1 = (1 / (0.5 * (trace - disc) + 1e-6)).reshape(-1)

        ur, vr, _ = func(img2, img1, pts, *args, **kwargs)
        ur, vr = ur.reshape(h, w), vr.reshape(h, w)
        coords = cp.stack([pts[:, 1] + v, pts[:, 0] + u])
        ur = image.map_coordinates(ur, coords, order=1, mode="mirror")
        vr = image.map_coordinates(vr, coords, order=1, mode="mirror")
        unc2 = cp.sqrt((u + ur) ** 2 + (v + vr) ** 2)

        uncs0.append(unc0[valid])
        uncs1.append(unc1[valid])
        uncs2.append(unc2[valid])
        del u, v, c, ur, vr, coords
        del truef, valid, ee, err, pts, x, y
        cp.get_default_memory_pool().free_all_blocks()

    epe, f1, nval = cp.array(epe), cp.array(f1), cp.array(nval)
    epe = cp.sum(epe * nval) / cp.sum(nval) if macro else cp.mean(epe)
    f1 = cp.sum(f1 * nval) / cp.sum(nval)

    names = [
        "PLK Covariance Trace (Ours)",
        "Structure Tensor Min Eigval",
        "Forward-Backward Consistency"
    ]
    auses, ccs = [], []

    for i, uncs in enumerate([uncs0, uncs1, uncs2]):
        cc = []
        for err, unc in zip(errs, uncs):
            rank_err = cp.argsort(cp.argsort(err)).astype(cp.float32)
            rank_unc = cp.argsort(cp.argsort(unc)).astype(cp.float32)
            cov = cp.cov(rank_err, rank_unc)
            cc.append(cov[0, 1] / cp.sqrt(cov[0, 0] * cov[1, 1]))
        ccs.append(cc := cp.mean(cp.array(cc)))

        ause = []
        for err0, unc in zip(errs, uncs):
            err = err0 / cp.mean(err0)
            orcerr = err[cp.argsort(-err)]
            orcauc = computeAUC(cp.sum(orcerr), orcerr, len(err), chunk)
            moderr = err[cp.argsort(-unc)]
            modauc = computeAUC(cp.sum(moderr), moderr, len(err), chunk)
            ause.append(modauc - orcauc)
        auses.append(ause := cp.mean(cp.array(ause)))
        print(f"{names[i]:>28s} | AUSE: {ause:.3f}, CC: {cc:.3f}")

### PARAMETER OPTIMIZATION

def searchParams(func, imgs, candidates, n_draws, macro):
    keys = list(candidates.keys())
    rng  = cp.random.default_rng(0)

    rows = []
    for _ in range(n_draws):
        cfg = {k: candidates[k][int(rng.integers(len(candidates[k])))]
               for k in keys}
        epe, f1 = run(func, imgs, macro=macro, **cfg)
        rows.append((tuple(cfg[k] for k in keys), float(epe), float(f1)))

    rows.sort(key=lambda r: r[0])
    fmt    = lambda v: f"{v:>9.3g}"
    label  = "AEPE" if macro else "FEPE"
    header = " ".join(f"{k:>9s}" for k in keys) + f" | {label:>9s} {'F1':>9s}"
    print("\n" + header)
    print("-" * len(header))
    for p, epe, f1 in rows:
        print(" ".join(fmt(v) for v in p)
              + f" | {epe:>9.3f} {f1:>9.3f}")
    return rows

def best(rows, metric):
    best = None
    for r in rows:
        if best is None or r[metric] < best[0]:
            best = (r[metric], r[0])
    return best

def topk(rows, metric, k):
    scored = []
    for r in rows:
        scored.append((r[metric], r[0]))
    scored.sort(key=lambda x: x[0])
    return scored[:k]

### DATA LOADING ###

!cp "/content/drive/My Drive/gpofe/mpisintel.zip" "/content"
!unzip -q /content/mpisintel.zip -d /content

imgs = [
    loadpair("mpisintel", i + 1, seq, "final")
    for seq in tqdm(listdir("/content/training/final"))
    for i in range(len(listdir(f"/content/training/flow/{seq}")))
]

!cp "/content/drive/My Drive/gpofe/kitti2015.zip" "/content"
!unzip -q /content/kitti2015.zip -d /content

imgs = [loadpair("kitti2015", i) for i in tqdm(range(200))]

### SINTEL (CLEAN) RESULTS ###

run(
    PFBC, imgs_val, macro=0,
    lvls=6, ksize=31, eps=0.1,
    gamma=1.0, sigma=10.0, fksize=9,
    fsigma=2.0, niter=5, tol=0.001
)

epe, f1, nval = [], [], []
for img1, img2, imgf in tqdm(imgs_u8):
    pred = cv.calcOpticalFlowFarneback(img1, img2, None, 0.5, 6, 31, 5, 9, 2, 0)
    ee, nv = computeEPE(imgf, pred)
    f1.append(computeF1(imgf, pred))
    epe.append(ee)
    nval.append(nv)
epe, d1, nval = np.array(epe), np.array(f1), np.array(nval)
epe = cp.sum(epe * nval) / cp.sum(nval)
f1 = cp.sum(f1 * nval) / cp.sum(nval)
print(f"\nAEPE: {epe:.3f}, F1: {f1:.3f}")

### SINTEL (FINAL) RESULTS ###

run(
    PFBC, imgs_val, macro=0,
    lvls=6, ksize=31, eps=0.1,
    gamma=1.0, sigma=10.0, fksize=9,
    fsigma=2.0, niter=5, tol=0.001
)

epe, f1, nval = [], [], []
for img1, img2, imgf in tqdm(imgs_u8):
    pred = cv.calcOpticalFlowFarneback(img1, img2, None, 0.5, 6, 31, 5, 9, 2, 0)
    ee, nv = computeEPE(imgf, pred)
    f1.append(computeF1(imgf, pred))
    epe.append(ee)
    nval.append(nv)
epe, d1, nval = np.array(epe), np.array(f1), np.array(nval)
epe = cp.sum(epe * nval) / cp.sum(nval)
f1 = cp.sum(f1 * nval) / cp.sum(nval)
print(f"\nAEPE: {epe:.3f}, F1: {f1:.3f}")

### KITTI RESULTS ###

run(
    PFBC, imgs_val, macro=0,
    lvls=5, ksize=15, eps=0.1,
    gamma=0.0, sigma=5.0, fksize=5,
    fsigma=1.2, niter=10, tol=0.0001
)

epe, f1, nval = [], [], []
for img1, img2, imgf in tqdm(imgs_u8):
    pred = cv.calcOpticalFlowFarneback(img1, img2, None, 0.5, 5, 15, 10, 5, 1.2, 0)
    ee, nv = computeEPE(imgf, pred)
    f1.append(computeF1(imgf, pred))
    epe.append(ee)
    nval.append(nv)
epe, d1, nval = np.array(epe), np.array(f1), np.array(nval)
epe = cp.mean(epe)
f1 = cp.sum(f1 * nval) / cp.sum(nval)
print(f"\nFEPE: {epe:.3f}, F1: {f1:.3f}")
