// cuda_sift.cu
// SIFT feature detection and descriptor computation on GPU.
// Pipeline: Gaussian pyramid → DoG → scale-space extrema →
//           subpixel refinement → orientation → 128-D descriptor.
//
// Outputs one binary file per image: <name>.sift
//   Header  : int32 n_keypoints
//   Keypoints: n × { float x, y, sigma, angle, response, int32 octave }
//   Descriptors: n × 128 × float32

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

// ─────────────────────────────── Parámetros SIFT ───────────────────────────
#define SIFT_OCTAVES         4
#define SIFT_INTERVALS       3                          // s
#define SIFT_GAUSS_PER_OCT   (SIFT_INTERVALS + 3)      // 6
#define SIFT_DOG_PER_OCT     (SIFT_INTERVALS + 2)      // 5
#define SIFT_SIGMA           1.6f
#define SIFT_CONTRAST_THR    0.04f
#define SIFT_EDGE_R          10.0f

#define MAX_KEYPOINTS        8192
#define DESCRIPTOR_DIMS      128

// orientación
#define ORI_BINS             36
#define ORI_RADIUS_FACTOR    3.0f
#define ORI_SIG_FACTOR       1.5f
#define ORI_PEAK_RATIO       0.8f

// descriptor
#define DESC_HIST_W          4
#define DESC_ORI_BINS        8
#define DESC_SCALE_FACTOR    3.0f
#define DESC_MAG_THR         0.2f

// tamaños de bloque
#define BLUR_BX              32
#define BLUR_BY              8
#define EXT_BX               16
#define EXT_BY               16

// ─────────────────────────────── Estructuras ───────────────────────────────
struct GpuKeypoint {
    float x, y;
    float sigma;
    float angle;
    float response;
    int   octave;
    int   layer;
    int   _pad;   // alineación a 32 bytes
};

// ═══════════════════════════════════════════════════════════════════════════
//  KERNEL 1 – Conversión a float [0,1] y upsample 2×
// ═══════════════════════════════════════════════════════════════════════════
__global__ void k_upsample2x(const unsigned char* __restrict__ src,
                              float* __restrict__ dst,
                              int src_w, int src_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_w = src_w * 2;
    int dst_h = src_h * 2;
    if (dx >= dst_w || dy >= dst_h) return;

    // bilinear interpolation from source
    float fx = (dx + 0.5f) * 0.5f - 0.5f;
    float fy = (dy + 0.5f) * 0.5f - 0.5f;
    int x0 = max(0, (int)floorf(fx));
    int y0 = max(0, (int)floorf(fy));
    int x1 = min(src_w - 1, x0 + 1);
    int y1 = min(src_h - 1, y0 + 1);
    float ax = fx - x0, ay = fy - y0;

    float v = (1-ax)*(1-ay)*src[y0*src_w+x0]
            + ax   *(1-ay)*src[y0*src_w+x1]
            + (1-ax)*ay   *src[y1*src_w+x0]
            + ax   * ay   *src[y1*src_w+x1];
    dst[dy * dst_w + dx] = v / 255.0f;
}

// ═══════════════════════════════════════════════════════════════════════════
//  KERNEL 2 – Desenfoque gaussiano separable (horizontal + vertical)
// ═══════════════════════════════════════════════════════════════════════════
// Tamaño de kernel gaussiano máximo
#define MAX_KSIZE 33
__constant__ float d_gauss_kernel[MAX_KSIZE];

__global__ void k_blur_h(const float* __restrict__ src, float* __restrict__ dst,
                          int w, int h, int ksize)
{
    extern __shared__ float smem[];   // (BLUR_BX + ksize-1) × BLUR_BY floats
    int half = ksize / 2;
    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * BLUR_BX + tx;
    int gy = blockIdx.y * BLUR_BY  + ty;

    int smem_w = BLUR_BX + ksize - 1;

    // carga con padding izquierdo
    int load_x = gx - half;
    for (int i = tx; i < smem_w; i += BLUR_BX) {
        int sx = blockIdx.x * BLUR_BX - half + i;
        sx = max(0, min(w - 1, sx));
        smem[ty * smem_w + i] = (gy < h) ? src[gy * w + sx] : 0.0f;
    }
    __syncthreads();

    if (gx >= w || gy >= h) return;

    float acc = 0.0f;
    for (int k = 0; k < ksize; k++)
        acc += smem[ty * smem_w + tx + k] * d_gauss_kernel[k];
    dst[gy * w + gx] = acc;
}

__global__ void k_blur_v(const float* __restrict__ src, float* __restrict__ dst,
                          int w, int h, int ksize)
{
    extern __shared__ float smem[];
    int half = ksize / 2;
    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * BLUR_BX + tx;
    int gy = blockIdx.y * BLUR_BY  + ty;

    int smem_h = BLUR_BY + ksize - 1;

    for (int j = ty; j < smem_h; j += BLUR_BY) {
        int sy = blockIdx.y * BLUR_BY - half + j;
        sy = max(0, min(h - 1, sy));
        smem[j * BLUR_BX + tx] = (gx < w) ? src[sy * w + gx] : 0.0f;
    }
    __syncthreads();

    if (gx >= w || gy >= h) return;

    float acc = 0.0f;
    for (int k = 0; k < ksize; k++)
        acc += smem[(ty + k) * BLUR_BX + tx] * d_gauss_kernel[k];
    dst[gy * w + gx] = acc;
}

// ═══════════════════════════════════════════════════════════════════════════
//  KERNEL 3 – Diferencia de Gaussianas (DoG)
// ═══════════════════════════════════════════════════════════════════════════
__global__ void k_dog(const float* __restrict__ g1, const float* __restrict__ g0,
                      float* __restrict__ dog, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dog[i] = g1[i] - g0[i];
}

// ═══════════════════════════════════════════════════════════════════════════
//  KERNEL 4 – Detección de extremos en escala-espacio
// ═══════════════════════════════════════════════════════════════════════════
__global__ void k_find_extrema(
    const float* __restrict__ dog0,
    const float* __restrict__ dog1,
    const float* __restrict__ dog2,
    int w, int h,
    GpuKeypoint* kpts, int* count,
    float contrast_thr, int octave, int layer,
    float sigma_layer)
{
    int gx = blockIdx.x * EXT_BX + threadIdx.x + 1;  // borde de 1 px
    int gy = blockIdx.y * EXT_BY + threadIdx.y + 1;
    if (gx >= w - 1 || gy >= h - 1) return;

    int idx = gy * w + gx;
    float v = dog1[idx];

    // filtro de contraste temprano
    if (fabsf(v) < 0.5f * contrast_thr) return;

    // comprobar máximo/mínimo respecto a 26 vecinos
    bool is_max = true, is_min = true;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int ni = (gy + dy) * w + (gx + dx);
            float n0 = dog0[ni], n1 = dog1[ni], n2 = dog2[ni];
            if (!(dy == 0 && dx == 0)) {
                if (n1 >= v) is_max = false;
                if (n1 <= v) is_min = false;
            }
            if (n0 >= v) is_max = false;
            if (n0 <= v) is_min = false;
            if (n2 >= v) is_max = false;
            if (n2 <= v) is_min = false;
        }
    }
    if (!is_max && !is_min) return;

    // eliminación de bordes: curvatura de Hessian 2D
    float dxx = dog1[gy*w + (gx+1)] + dog1[gy*w + (gx-1)] - 2*v;
    float dyy = dog1[(gy+1)*w + gx] + dog1[(gy-1)*w + gx] - 2*v;
    float dxy = 0.25f * (dog1[(gy+1)*w+(gx+1)] - dog1[(gy+1)*w+(gx-1)]
                       - dog1[(gy-1)*w+(gx+1)] + dog1[(gy-1)*w+(gx-1)]);
    float tr  = dxx + dyy;
    float det = dxx * dyy - dxy * dxy;
    if (det <= 0.0f) return;
    float edge_ratio = tr * tr / det;
    float edge_thr = (SIFT_EDGE_R + 1.0f) * (SIFT_EDGE_R + 1.0f) / SIFT_EDGE_R;
    if (edge_ratio >= edge_thr) return;

    int pos = atomicAdd(count, 1);
    if (pos >= MAX_KEYPOINTS) { atomicSub(count, 1); return; }

    GpuKeypoint& kp = kpts[pos];
    kp.x       = (float)gx;
    kp.y       = (float)gy;
    kp.sigma   = sigma_layer;
    kp.angle   = 0.0f;
    kp.response = fabsf(v);
    kp.octave  = octave;
    kp.layer   = layer;
}

// ═══════════════════════════════════════════════════════════════════════════
//  KERNEL 5 – Gradientes de magnitud y orientación
// ═══════════════════════════════════════════════════════════════════════════
__global__ void k_gradient(const float* __restrict__ img,
                            float* __restrict__ mag,
                            float* __restrict__ ori,
                            int w, int h)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= w || gy >= h) { return; }

    int idx = gy * w + gx;
    float dx = 0.0f, dy = 0.0f;

    if (gx > 0 && gx < w-1)
        dx = 0.5f * (img[gy*w + gx+1] - img[gy*w + gx-1]);
    if (gy > 0 && gy < h-1)
        dy = 0.5f * (img[(gy+1)*w + gx] - img[(gy-1)*w + gx]);

    mag[idx] = sqrtf(dx*dx + dy*dy);
    ori[idx] = atan2f(dy, dx);   // [-π, π]
}

// ═══════════════════════════════════════════════════════════════════════════
//  KERNEL 6 – Asignación de orientación dominante (1 hilo por keypoint)
// ═══════════════════════════════════════════════════════════════════════════
__global__ void k_orientation(
    const float* __restrict__ mag,
    const float* __restrict__ ori,
    int w, int h,
    GpuKeypoint* kpts, int n_kpts)
{
    int kid = blockIdx.x * blockDim.x + threadIdx.x;
    if (kid >= n_kpts) return;

    GpuKeypoint& kp = kpts[kid];
    float cx = kp.x, cy = kp.y;
    float sigma = kp.sigma;
    float radius = ORI_RADIUS_FACTOR * sigma;
    float weight_sigma2_inv = 0.5f / (ORI_SIG_FACTOR * sigma * ORI_SIG_FACTOR * sigma);

    float hist[ORI_BINS] = {0};

    int r = (int)ceilf(radius);
    for (int dy = -r; dy <= r; dy++) {
        int py = (int)(cy + dy + 0.5f);
        if (py < 1 || py >= h-1) continue;
        for (int dx = -r; dx <= r; dx++) {
            int px = (int)(cx + dx + 0.5f);
            if (px < 1 || px >= w-1) continue;
            float dist2 = (float)(dx*dx + dy*dy);
            if (dist2 > radius * radius) continue;

            float w_gauss = expf(-dist2 * weight_sigma2_inv);
            float m = mag[py * w + px];
            float o = ori[py * w + px];                    // [-π, π]
            float deg = o * (180.0f / CUDART_PI_F);
            if (deg < 0) deg += 360.0f;

            int bin = (int)(deg * ORI_BINS / 360.0f) % ORI_BINS;
            hist[bin] += w_gauss * m;
        }
    }

    // suavizado del histograma (6 iteraciones)
    for (int iter = 0; iter < 6; iter++) {
        float prev = hist[ORI_BINS - 1];
        for (int b = 0; b < ORI_BINS; b++) {
            float tmp = hist[b];
            hist[b] = 0.25f * prev + 0.5f * hist[b]
                    + 0.25f * hist[(b + 1) % ORI_BINS];
            prev = tmp;
        }
    }

    // pico máximo
    float peak = 0.0f;
    int   peak_bin = 0;
    for (int b = 0; b < ORI_BINS; b++) {
        if (hist[b] > peak) { peak = hist[b]; peak_bin = b; }
    }

    // interpolación parabólica del pico
    float left  = hist[(peak_bin - 1 + ORI_BINS) % ORI_BINS];
    float right = hist[(peak_bin + 1) % ORI_BINS];
    float interp = 0.5f * (left - right) / (left - 2*peak + right + 1e-10f);
    float angle_deg = (peak_bin + 0.5f + interp) * 360.0f / ORI_BINS;
    kp.angle = angle_deg * (CUDART_PI_F / 180.0f);
}

// ═══════════════════════════════════════════════════════════════════════════
//  KERNEL 7 – Descriptor SIFT 128D (1 hilo por keypoint)
// ═══════════════════════════════════════════════════════════════════════════
__global__ void k_descriptor(
    const float* __restrict__ mag,
    const float* __restrict__ ori,
    int w, int h,
    const GpuKeypoint* __restrict__ kpts, int n_kpts,
    float* __restrict__ descs)    // n_kpts × 128
{
    int kid = blockIdx.x * blockDim.x + threadIdx.x;
    if (kid >= n_kpts) return;

    const GpuKeypoint& kp = kpts[kid];
    float* desc = descs + kid * DESCRIPTOR_DIMS;

    float cx = kp.x, cy = kp.y;
    float sigma = kp.sigma;
    float cos_a = cosf(-kp.angle), sin_a = sinf(-kp.angle);

    float hist_width = DESC_SCALE_FACTOR * sigma;
    int   radius     = (int)(hist_width * (DESC_HIST_W + 1) * 0.5f * sqrtf(2.0f) + 0.5f);

    float bins[DESC_HIST_W][DESC_HIST_W][DESC_ORI_BINS] = {{{0}}};

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            // rotación al sistema de referencia del keypoint
            float rx = cos_a * dx - sin_a * dy;
            float ry = sin_a * dx + cos_a * dy;

            // coordenadas en la rejilla [0, DESC_HIST_W)
            float bx = rx / hist_width + DESC_HIST_W * 0.5f - 0.5f;
            float by = ry / hist_width + DESC_HIST_W * 0.5f - 0.5f;

            if (bx <= -1.0f || bx >= DESC_HIST_W || by <= -1.0f || by >= DESC_HIST_W)
                continue;

            int py = (int)(cy + dy + 0.5f);
            int px = (int)(cx + dx + 0.5f);
            if (px < 1 || px >= w-1 || py < 1 || py >= h-1) continue;

            // peso gaussiano
            float wx  = bx - floorf(bx);
            float wy  = by - floorf(by);
            float dist2 = (rx*rx + ry*ry) / (2.0f * (0.5f*DESC_HIST_W*hist_width)
                                                   * (0.5f*DESC_HIST_W*hist_width));
            float w_gauss = expf(-dist2);

            float m = mag[py * w + px];
            float o = ori[py * w + px] - kp.angle;
            if (o < 0) o += 2.0f * CUDART_PI_F;
            if (o >= 2.0f * CUDART_PI_F) o -= 2.0f * CUDART_PI_F;

            float obin = o * (DESC_ORI_BINS / (2.0f * CUDART_PI_F));
            float wo   = obin - floorf(obin);
            int   o0   = ((int)obin) % DESC_ORI_BINS;
            int   o1   = (o0 + 1)   % DESC_ORI_BINS;

            float val = m * w_gauss;

            // interpolación trilineal en (bx, by, obin)
            int bxi = (int)bx, byi = (int)by;
            for (int bj = 0; bj <= 1; bj++) {
                int bidy = byi + bj;
                if (bidy < 0 || bidy >= DESC_HIST_W) continue;
                float wy_c = bj == 0 ? (1.0f - wy) : wy;
                for (int bi = 0; bi <= 1; bi++) {
                    int bidx = bxi + bi;
                    if (bidx < 0 || bidx >= DESC_HIST_W) continue;
                    float wx_c = bi == 0 ? (1.0f - wx) : wx;
                    bins[bidy][bidx][o0] += val * wy_c * wx_c * (1.0f - wo);
                    bins[bidy][bidx][o1] += val * wy_c * wx_c * wo;
                }
            }
        }
    }

    // aplanar a vector de 128
    int k = 0;
    float norm2 = 0.0f;
    for (int i = 0; i < DESC_HIST_W; i++)
        for (int j = 0; j < DESC_HIST_W; j++)
            for (int o = 0; o < DESC_ORI_BINS; o++) {
                desc[k] = bins[i][j][o];
                norm2 += desc[k] * desc[k];
                k++;
            }

    // normalización L2
    float inv_norm = rsqrtf(norm2 + 1e-10f);
    for (int i = 0; i < DESCRIPTOR_DIMS; i++)
        desc[i] *= inv_norm;

    // clamp y renormalización
    norm2 = 0.0f;
    for (int i = 0; i < DESCRIPTOR_DIMS; i++) {
        desc[i] = fminf(desc[i], DESC_MAG_THR);
        norm2  += desc[i] * desc[i];
    }
    inv_norm = rsqrtf(norm2 + 1e-10f);
    for (int i = 0; i < DESCRIPTOR_DIMS; i++)
        desc[i] = fminf(roundf(desc[i] * inv_norm * 512.0f), 255.0f);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Utilidades: kernel gaussiano, gestión de memoria
// ═══════════════════════════════════════════════════════════════════════════
static int gaussian_ksize(float sigma) {
    int k = (int)(6.0f * sigma + 1.0f);
    if (k % 2 == 0) k++;
    if (k < 3) k = 3;
    if (k > MAX_KSIZE) k = MAX_KSIZE;
    return k;
}

static void upload_gaussian_kernel(float sigma, int ksize) {
    int half = ksize / 2;
    std::vector<float> kdata(ksize);
    double sum = 0.0;
    for (int i = 0; i < ksize; i++) {
        float x = (float)(i - half);
        kdata[i] = expf(-x*x / (2.0f*sigma*sigma));
        sum += kdata[i];
    }
    for (int i = 0; i < ksize; i++) kdata[i] /= (float)sum;
    cudaMemcpyToSymbol(d_gauss_kernel, kdata.data(), ksize * sizeof(float));
}

// Aplica desenfoque gaussiano in-place: src → blur (tmp es buffer temporal)
static void gpu_blur(float* src, float* tmp, float* dst,
                     int w, int h, float sigma)
{
    int ksize = gaussian_ksize(sigma);
    upload_gaussian_kernel(sigma, ksize);

    dim3 block(BLUR_BX, BLUR_BY);
    dim3 grid((w + BLUR_BX - 1) / BLUR_BX, (h + BLUR_BY - 1) / BLUR_BY);

    size_t smem_h = (size_t)(BLUR_BX + ksize - 1) * BLUR_BY * sizeof(float);
    size_t smem_v = (size_t)(BLUR_BY + ksize - 1) * BLUR_BX * sizeof(float);

    k_blur_h<<<grid, block, smem_h>>>(src, tmp, w, h, ksize);
    k_blur_v<<<grid, block, smem_v>>>(tmp, dst, w, h, ksize);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Función principal de host: cuda_sift()
// ═══════════════════════════════════════════════════════════════════════════
struct SiftResult {
    std::vector<GpuKeypoint> keypoints;
    std::vector<float>       descriptors;   // n × 128 floats
};

static SiftResult cuda_sift(const cv::Mat& gray_img)
{
    SiftResult result;

    int src_w = gray_img.cols;
    int src_h = gray_img.rows;
    int up_w  = src_w * 2;
    int up_h  = src_h * 2;

    // ── Subir imagen a GPU ──────────────────────────────────────────────────
    unsigned char* d_src;
    cudaMalloc(&d_src, src_w * src_h);
    cudaMemcpy(d_src, gray_img.data, src_w * src_h, cudaMemcpyHostToDevice);

    // ── Upsample 2× de la imagen base ──────────────────────────────────────
    float* d_base;
    cudaMalloc(&d_base, up_w * up_h * sizeof(float));
    {
        dim3 blk(16, 16), grd((up_w+15)/16, (up_h+15)/16);
        k_upsample2x<<<grd, blk>>>(d_src, d_base, src_w, src_h);
    }
    cudaFree(d_src);

    // Desenfoque inicial de la base para alcanzar sigma = SIFT_SIGMA
    // assumed camera blur = 0.5, upsampled doubles it to 1.0
    float base_sigma = sqrtf(SIFT_SIGMA * SIFT_SIGMA - 1.0f);
    {
        float* tmp; cudaMalloc(&tmp, up_w * up_h * sizeof(float));
        float* blurred; cudaMalloc(&blurred, up_w * up_h * sizeof(float));
        gpu_blur(d_base, tmp, blurred, up_w, up_h, base_sigma);
        cudaFree(d_base);
        cudaFree(tmp);
        d_base = blurred;
    }

    // ── Buffers para keypoints (conteo en GPU) ─────────────────────────────
    GpuKeypoint* d_kpts;
    int*         d_count;
    cudaMalloc(&d_kpts,  MAX_KEYPOINTS * sizeof(GpuKeypoint));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    // ── Pirámide gaussiana + DoG por octava ───────────────────────────────
    float k_factor = powf(2.0f, 1.0f / SIFT_INTERVALS);

    // Sigmas incrementales entre escalas consecutivas de la misma octava
    std::vector<float> sigmas(SIFT_GAUSS_PER_OCT);
    sigmas[0] = SIFT_SIGMA;
    for (int i = 1; i < SIFT_GAUSS_PER_OCT; i++) {
        float sigma_prev = SIFT_SIGMA * powf(k_factor, (float)(i - 1));
        float sigma_curr = sigma_prev * k_factor;
        sigmas[i] = sqrtf(sigma_curr * sigma_curr - sigma_prev * sigma_prev);
    }

    // punteros a las imágenes de la pirámide para la octava actual
    float* gauss[SIFT_GAUSS_PER_OCT];
    float* dog  [SIFT_DOG_PER_OCT];
    float* tmp_blur;

    float* oct_input = d_base;

    for (int oct = 0; oct < SIFT_OCTAVES; oct++) {
        int oct_w = up_w >> oct;
        int oct_h = up_h >> oct;
        int oct_n = oct_w * oct_h;

        // — Si no es la primera octava, hacer downsample 2× desde gauss[SIFT_INTERVALS]
        if (oct > 0) {
            // gauss[SIFT_INTERVALS] de la octava anterior es el input de esta octava
            float* d_down;
            cudaMalloc(&d_down, oct_n * sizeof(float));
            // downsample simple: tomar pixel (2i, 2j) de la imagen anterior
            // La imagen anterior tiene tamaño oct_w*2 × oct_h*2
            // Usamos un kernel inline en lambda; como CUDA no soporta lambdas en kernels,
            // reutilizamos k_upsample2x al revés mediante un kernel ad-hoc.
            // Por simplicidad lo hacemos en CPU en este caso (la imagen ya está en GPU).
            int prev_w = oct_w * 2, prev_h = oct_h * 2;
            std::vector<float> h_prev(prev_w * prev_h);
            cudaMemcpy(h_prev.data(), oct_input,
                       prev_w * prev_h * sizeof(float), cudaMemcpyDeviceToHost);
            std::vector<float> h_down(oct_n);
            for (int j = 0; j < oct_h; j++)
                for (int i = 0; i < oct_w; i++)
                    h_down[j * oct_w + i] = h_prev[(2*j) * prev_w + (2*i)];
            cudaMemcpy(d_down, h_down.data(), oct_n * sizeof(float), cudaMemcpyHostToDevice);
            // Liberar el gauss[SIFT_INTERVALS] de la octava previa (ya volcado a CPU).
            cudaFree(oct_input);
            oct_input = d_down;
        }

        // — Alojar pirámide gaussiana
        for (int s = 0; s < SIFT_GAUSS_PER_OCT; s++)
            cudaMalloc(&gauss[s], oct_n * sizeof(float));
        for (int s = 0; s < SIFT_DOG_PER_OCT; s++)
            cudaMalloc(&dog[s],   oct_n * sizeof(float));
        cudaMalloc(&tmp_blur, oct_n * sizeof(float));

        // gauss[0] = entrada (ya con sigma = SIFT_SIGMA para esta octava)
        cudaMemcpy(gauss[0], oct_input, oct_n * sizeof(float), cudaMemcpyDeviceToDevice);

        // — Construir pirámide gaussiana
        for (int s = 1; s < SIFT_GAUSS_PER_OCT; s++)
            gpu_blur(gauss[s-1], tmp_blur, gauss[s], oct_w, oct_h, sigmas[s]);

        cudaFree(tmp_blur);

        // — Construir pirámide DoG
        for (int s = 0; s < SIFT_DOG_PER_OCT; s++) {
            int n = oct_n;
            k_dog<<<(n + 255) / 256, 256>>>(gauss[s+1], gauss[s], dog[s], n);
        }

        // — Detectar extremos en las capas interiores (1 … SIFT_INTERVALS)
        for (int s = 1; s <= SIFT_INTERVALS; s++) {
            float sigma_s = SIFT_SIGMA * powf(k_factor, (float)s) * (float)(1 << oct);
            float cthr    = SIFT_CONTRAST_THR / SIFT_INTERVALS;

            dim3 blk(EXT_BX, EXT_BY);
            dim3 grd((oct_w + EXT_BX - 1) / EXT_BX, (oct_h + EXT_BY - 1) / EXT_BY);

            k_find_extrema<<<grd, blk>>>(
                dog[s-1], dog[s], dog[s+1],
                oct_w, oct_h,
                d_kpts, d_count,
                cthr, oct, s, sigma_s);
        }

        // Para la siguiente octava, la entrada es gauss[SIFT_INTERVALS]
        if (oct < SIFT_OCTAVES - 1) {
            if (oct > 0) cudaFree(oct_input);  // liberar el downsampled anterior
            oct_input = gauss[SIFT_INTERVALS];
            // no liberar oct_input todavía; se liberará al inicio del próximo oct
        } else {
            if (oct > 0) cudaFree(oct_input);
            oct_input = nullptr;   // evitar double-free posterior
        }

        for (int s = 0; s < SIFT_GAUSS_PER_OCT; s++)
            if (!(oct < SIFT_OCTAVES - 1 && s == SIFT_INTERVALS))
                cudaFree(gauss[s]);
        for (int s = 0; s < SIFT_DOG_PER_OCT; s++)
            cudaFree(dog[s]);
    }
    if (oct_input && oct_input != d_base) cudaFree(oct_input);
    cudaFree(d_base);

    // ── Número de keypoints detectados ────────────────────────────────────
    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_count);

    if (h_count == 0) {
        cudaFree(d_kpts);
        return result;
    }

    // ── Recalcular gradientes sobre la imagen original redimensionada ──────
    // Usamos la imagen de entrada en float (convertida sin upsample para grad)
    float* d_img_f;
    cudaMalloc(&d_img_f, src_w * src_h * sizeof(float));
    {
        unsigned char* d_tmp_u8;
        cudaMalloc(&d_tmp_u8, src_w * src_h);
        cudaMemcpy(d_tmp_u8, gray_img.data, src_w * src_h, cudaMemcpyHostToDevice);
        // convertir uint8 a float normalizado reutilizando k_upsample2x con escala 1
        // — alternativa simple: hacerlo en CPU
        cudaFree(d_tmp_u8);
    }
    {
        std::vector<float> h_f(src_w * src_h);
        for (int i = 0; i < src_w * src_h; i++)
            h_f[i] = gray_img.data[i] / 255.0f;
        cudaMemcpy(d_img_f, h_f.data(), src_w * src_h * sizeof(float), cudaMemcpyHostToDevice);
    }

    float* d_mag, *d_ori;
    cudaMalloc(&d_mag, up_w * up_h * sizeof(float));
    cudaMalloc(&d_ori, up_w * up_h * sizeof(float));

    // reconstruir imagen upsampled para los gradientes del descriptor
    float* d_up_f;
    cudaMalloc(&d_up_f, up_w * up_h * sizeof(float));
    {
        unsigned char* d_u8;
        cudaMalloc(&d_u8, src_w * src_h);
        cudaMemcpy(d_u8, gray_img.data, src_w * src_h, cudaMemcpyHostToDevice);
        dim3 blk(16,16), grd((up_w+15)/16, (up_h+15)/16);
        k_upsample2x<<<grd, blk>>>(d_u8, d_up_f, src_w, src_h);
        cudaFree(d_u8);
    }

    {
        dim3 blk(16,16), grd((up_w+15)/16, (up_h+15)/16);
        k_gradient<<<grd, blk>>>(d_up_f, d_mag, d_ori, up_w, up_h);
    }
    cudaFree(d_up_f);
    cudaFree(d_img_f);

    // ── Orientación ─────────────────────────────────────────────────────────
    {
        int tpb = 128;
        k_orientation<<<(h_count + tpb - 1) / tpb, tpb>>>(
            d_mag, d_ori, up_w, up_h, d_kpts, h_count);
    }

    // ── Descriptores ────────────────────────────────────────────────────────
    float* d_descs;
    cudaMalloc(&d_descs, h_count * DESCRIPTOR_DIMS * sizeof(float));
    {
        int tpb = 64;
        k_descriptor<<<(h_count + tpb - 1) / tpb, tpb>>>(
            d_mag, d_ori, up_w, up_h, d_kpts, h_count, d_descs);
    }

    cudaFree(d_mag);
    cudaFree(d_ori);

    // ── Copiar resultados a host ─────────────────────────────────────────────
    result.keypoints.resize(h_count);
    result.descriptors.resize(h_count * DESCRIPTOR_DIMS);

    cudaMemcpy(result.keypoints.data(),  d_kpts,
               h_count * sizeof(GpuKeypoint), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.descriptors.data(), d_descs,
               h_count * DESCRIPTOR_DIMS * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_kpts);
    cudaFree(d_descs);

    // Las coordenadas de los keypoints están en espacio upsampled (2×);
    // convertirlas al espacio original.
    for (auto& kp : result.keypoints) {
        kp.x     /= 2.0f;
        kp.y     /= 2.0f;
        kp.sigma /= 2.0f;
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Escritura del archivo binario .sift
// ═══════════════════════════════════════════════════════════════════════════
static void write_sift_binary(const std::string& path, const SiftResult& res)
{
    std::ofstream f(path, std::ios::binary);
    int32_t n = (int32_t)res.keypoints.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (const auto& kp : res.keypoints) {
        f.write(reinterpret_cast<const char*>(&kp.x),       sizeof(float));
        f.write(reinterpret_cast<const char*>(&kp.y),       sizeof(float));
        f.write(reinterpret_cast<const char*>(&kp.sigma),   sizeof(float));
        f.write(reinterpret_cast<const char*>(&kp.angle),   sizeof(float));
        f.write(reinterpret_cast<const char*>(&kp.response),sizeof(float));
        int32_t oct = kp.octave;
        f.write(reinterpret_cast<const char*>(&oct),        sizeof(int32_t));
    }
    if (!res.descriptors.empty())
        f.write(reinterpret_cast<const char*>(res.descriptors.data()),
                n * DESCRIPTOR_DIMS * sizeof(float));
}

// ═══════════════════════════════════════════════════════════════════════════
//  main()
// ═══════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[])
{
    namespace fs = std::filesystem;

    std::string dir = (argc >= 2) ? argv[1] : "../data/procesadas_filtros_cuda";

    // Recopilar imágenes
    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());

    if (files.size() < 2) {
        std::cerr << "cuda_sift: se necesitan al menos 2 imágenes en " << dir << "\n";
        return 1;
    }

    // Verificar GPU
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        std::cerr << "cuda_sift: no se encontró GPU CUDA.\n";
        return 2;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "cuda_sift: usando GPU \"" << prop.name << "\"\n";
    std::cout << "cuda_sift: procesando " << files.size() << " imágenes...\n";

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    std::string out_dir = "../data/procesadas_filtros_cuda";
    fs::create_directories(out_dir);

    for (const auto& fpath : files) {
        cv::Mat img = cv::imread(fpath, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "  [warn] no se pudo cargar: " << fpath << "\n";
            continue;
        }

        SiftResult res = cuda_sift(img);
        std::cout << "  " << fs::path(fpath).filename().string()
                  << " → " << res.keypoints.size() << " keypoints\n";

        std::string stem = fs::path(fpath).stem().string();
        std::string out = out_dir + "/" + stem + ".sift";
        write_sift_binary(out, res);
    }

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    float t_cuda = ms / 1000.0f;
    std::cout << "cuda_sift: completado en " << t_cuda << " s\n";

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    FILE *csv = fopen("../results/tiempos.csv", "a");
    if (csv) {
        fprintf(csv, "sift_cuda,%.6f,%d\n", t_cuda, (int)files.size());
        fclose(csv);
        std::cout << "Tiempo CUDA exportado a ../results/tiempos.csv" << std::endl;
    }

    return 0;
}