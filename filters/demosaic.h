#ifndef DEMOSAIC_FILTER_H
#define DEMOSAIC_FILTER_H

#include "../image.h"
#include "util.h"

stbi_uc* demosaic(stbi_uc* input_image, int width, int height, int channels);
__global__ void demosaicKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels);
void demosaic(uint8_t* raw_buf, uint8_t* rgb_buf, uint32_t width, uint32_t height);


stbi_uc* demosaic(stbi_uc* input_image, int width, int height, int channels) {
    int image_size_input = width * height * sizeof(stbi_uc);
    int image_size_output = 3 * width * height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_output_image;
    stbi_uc* h_output_image = (stbi_uc*)malloc(image_size_output);

    cudaMallocManaged(&d_input_image, image_size_input);
    cudaMallocManaged(&d_output_image, image_size_output);
    cudaMemcpy(d_input_image, input_image, image_size_input, cudaMemcpyHostToDevice);

    // 使用二維配置
    dim3 threads(16, 16); // 每個區塊 16x16 的執行緒
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    printf("Blocks: %d x %d, Threads: %d x %d\n", blocks.x, blocks.y, threads.x, threads.y);

    demosaicKernel<<<blocks, threads>>>(d_input_image, d_output_image, width, height, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_image, d_output_image, image_size_output, cudaMemcpyDeviceToHost);
    cudaFree(d_input_image);
    cudaFree(d_output_image);

    return h_output_image;
}

__global__ void demosaicKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels) {
    // 計算當前執行緒的座標
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 邊界條件檢查
    if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2) {
        return;
    }

    int idx = y * width + x;
    int R = 0, G = 0, B = 0;

    // Malvar interpolation for RGGB pattern
    if ((y % 2 == 0) && (x % 2 == 0)) {  // Red pixel
        R = input_image[idx];
        G = (4 * input_image[idx] - input_image[idx - 2 * width] - input_image[idx - 2] -
             input_image[idx + 2 * width] - input_image[idx + 2] +
             2 * (input_image[idx - width] + input_image[idx + width] +
                  input_image[idx - 1] + input_image[idx + 1])) >> 3;
        B = (6 * input_image[idx] -
             ((3 * (input_image[idx - 2 * width] + input_image[idx - 2] +
                    input_image[idx + 2 * width] + input_image[idx + 2])) >>
              1) +
             2 * (input_image[idx - width - 1] + input_image[idx - width + 1] +
                  input_image[idx + width - 1] + input_image[idx + width + 1])) >>
            3;
    } else if ((y % 2 == 0) && (x % 2 == 1)) {  // Green pixel on Red row
        R = (5 * input_image[idx] - input_image[idx - 2] - input_image[idx - width - 1] -
             input_image[idx + width - 1] - input_image[idx - width + 1] -
             input_image[idx + width + 1] - input_image[idx + 2] +
             ((input_image[idx - 2 * width] + input_image[idx + 2 * width]) >> 1) +
             4 * (input_image[idx - 1] + input_image[idx + 1])) >>
            3;
        G = input_image[idx];
        B = (5 * input_image[idx] - input_image[idx - 2 * width] - input_image[idx - width - 1] -
             input_image[idx - width + 1] - input_image[idx + 2 * width] -
             input_image[idx + width - 1] - input_image[idx + width + 1] +
             ((input_image[idx - 2] + input_image[idx + 2]) >> 1) +
             4 * (input_image[idx - width] + input_image[idx + width])) >>
            3;
    } else if ((y % 2 == 1) && (x % 2 == 0)) {  // Green pixel on Blue row
        R = (5 * input_image[idx] - input_image[idx - 2 * width] - input_image[idx - width - 1] -
             input_image[idx - width + 1] - input_image[idx + 2 * width] -
             input_image[idx + width - 1] - input_image[idx + width + 1] +
             ((input_image[idx - 2] - input_image[idx + 2]) >> 1) +
             4 * (input_image[idx - width] + input_image[idx + width])) >>
            3;
        G = input_image[idx];
        B = (5 * input_image[idx] - input_image[idx - 2] - input_image[idx - width - 1] -
             input_image[idx + width - 1] - input_image[idx - width + 1] -
             input_image[idx + width + 1] - input_image[idx + 2] +
             ((input_image[idx - 2 * width] + input_image[idx + 2 * width]) >> 1) +
             4 * (input_image[idx - 1] + input_image[idx + 1])) >>
            3;
    } else {  // Blue pixel
        R = (6 * input_image[idx] -
             ((3 * (input_image[idx - 2 * width] + input_image[idx - 2] +
                    input_image[idx + 2 * width] + input_image[idx + 2])) >>
              1) +
             2 * (input_image[idx - width - 1] + input_image[idx - width + 1] +
                  input_image[idx + width - 1] + input_image[idx + width + 1])) >>
            3;
        G = (4 * input_image[idx] - input_image[idx - 2 * width] - input_image[idx - 2] -
             input_image[idx + 2 * width] - input_image[idx + 2] +
             2 * (input_image[idx - width] + input_image[idx + width] +
                  input_image[idx - 1] + input_image[idx + 1])) >>
            3;
        B = input_image[idx];
    }

    // Clip values to [0, 255]
    R = max(0, min(255, R));
    G = max(0, min(255, G));
    B = max(0, min(255, B));

    // 將結果存儲到全局記憶體
    output_image[3 * idx] = R;
    output_image[3 * idx + 1] = G;
    output_image[3 * idx + 2] = B;
}


void demosaic(uint8_t* raw_buf, uint8_t* rgb_buf, uint32_t width, uint32_t height) {
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            int idx = i * width + j;
            int R = 0, G = 0, B = 0;
            if (i < 2 || i >= height - 2 || j < 2 || j >= width - 2) {
                rgb_buf[3 * idx] = 0;
                rgb_buf[3 * idx + 1] = 0;
                rgb_buf[3 * idx + 2] = 0;
                continue;
            }

            // Malvar interpolation for RGGB pattern
            if ((i % 2 == 0) && (j % 2 == 0)) {  // Red pixel
                R = raw_buf[idx];
                G = (4 * raw_buf[idx] - raw_buf[idx - 2 * width] - raw_buf[idx - 2] - raw_buf[idx + 2 * width] - raw_buf[idx + 2] +
                    2 * (raw_buf[idx + width] + raw_buf[idx + 1] + raw_buf[idx - width] + raw_buf[idx - 1])) >> 3;
                B = (6 * raw_buf[idx] - ((3 * (raw_buf[idx - 2 * width] + raw_buf[idx - 2] + raw_buf[idx + 2 * width] + raw_buf[idx + 2])) >> 1) +
                    2 * (raw_buf[idx - width - 1] + raw_buf[idx - width + 1] + raw_buf[idx + width - 1] + raw_buf[idx + width + 1])) >> 3;
            } else if ((i % 2 == 0) && (j % 2 == 1)) {  // Green pixel on Red row
                R = (5 * raw_buf[idx] - raw_buf[idx - 2] - raw_buf[idx - width - 1] - raw_buf[idx + width - 1] -
                    raw_buf[idx - width + 1] - raw_buf[idx + width + 1] - raw_buf[idx + 2] +
                    ((raw_buf[idx - 2 * width] + raw_buf[idx + 2 * width]) >> 1) + 4 * (raw_buf[idx - 1] + raw_buf[idx + 1])) >> 3;
                G = raw_buf[idx];
                B = (5 * raw_buf[idx] - raw_buf[idx - 2 * width] - raw_buf[idx - width - 1] - raw_buf[idx - width + 1] -
                    raw_buf[idx + 2 * width] - raw_buf[idx + width - 1] - raw_buf[idx + width + 1] +
                    ((raw_buf[idx - 2] + raw_buf[idx + 2]) >> 1) + 4 * (raw_buf[idx - width] + raw_buf[idx + width])) >> 3;
            } else if ((i % 2 == 1) && (j % 2 == 0)) {  // Green pixel on Blue row
                R = (5 * raw_buf[idx] - raw_buf[idx - 2 * width] - raw_buf[idx - width - 1] - raw_buf[idx - width + 1] -
                    raw_buf[idx + 2 * width] - raw_buf[idx + width - 1] - raw_buf[idx + width + 1] +
                    ((raw_buf[idx - 2] - raw_buf[idx + 2]) >> 1) + 4 * (raw_buf[idx - width] + raw_buf[idx + width])) >> 3;
                G = raw_buf[idx];
                B = (5 * raw_buf[idx] - raw_buf[idx - 2] - raw_buf[idx - width - 1] - raw_buf[idx + width - 1] - raw_buf[idx - width + 1] -
                    raw_buf[idx + width + 1] - raw_buf[idx + 2] + ((raw_buf[idx - 2 * width] + raw_buf[idx + 2 * width]) >> 1) +
                    4 * (raw_buf[idx - 1] + raw_buf[idx + 1])) >> 3;
            } else {  // Blue pixel
                R = (6 * raw_buf[idx] - ((3 * (raw_buf[idx - 2 * width] + raw_buf[idx - 2] + raw_buf[idx + 2 * width] + raw_buf[idx + 2])) >> 1) +
                    2 * (raw_buf[idx - width - 1] + raw_buf[idx - width + 1] + raw_buf[idx + width - 1] + raw_buf[idx + width + 1])) >> 3;
                G = (4 * raw_buf[idx] - raw_buf[idx - 2 * width] - raw_buf[idx - 2] - raw_buf[idx + 2 * width] - raw_buf[idx + 2] +
                    2 * (raw_buf[idx + width] + raw_buf[idx + 1] + raw_buf[idx - width] + raw_buf[idx - 1])) >> 3;
                B = raw_buf[idx];
            }

            // Clip values to [0, 255]
            R = clamp_cpu(R, 0, 255);
            G = clamp_cpu(G, 0, 255);
            B = clamp_cpu(B, 0, 255);

            // Store RGB values
            rgb_buf[3 * idx] = (uint8_t)(R);
            rgb_buf[3 * idx + 1] = (uint8_t)(G);
            rgb_buf[3 * idx + 2] = (uint8_t)(B);
        }
    }
}

#endif

