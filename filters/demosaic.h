#ifndef DEMOSAIC_FILTER_H
#define DEMOSAIC_FILTER_H

#include "../image.h"
#include "util.h"

stbi_uc* demosaic(stbi_uc* input_image, int width, int height, int channels);
__global__ void demosaicKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads);

stbi_uc* demosaic(stbi_uc* input_image, int width, int height, int channels) {
    int image_size_input = 1 * width * height * sizeof(stbi_uc);
    int image_size_output = 3 * width * height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_output_image;
    stbi_uc* h_output_image = (stbi_uc*)malloc(image_size_output);
    cudaMallocManaged(&d_input_image, image_size_input);
    cudaMallocManaged(&d_output_image, image_size_output);
    cudaMemcpy(d_input_image, input_image, image_size_input, cudaMemcpyHostToDevice);

    int total_threads = (width - 1) * (height - 1);
    int threads = min(THREADS_PER_BLOCK, total_threads);
    int blocks = (threads == total_threads) ? 1 : total_threads / THREADS_PER_BLOCK;

    printf("Blocks %d, threads %d\n", blocks, threads);
    demosaicKernel<<<blocks, threads>>>(d_input_image, d_output_image, width, height, channels, total_threads);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_image, d_output_image, image_size_output, cudaMemcpyDeviceToHost);
    cudaFree(d_input_image);
    cudaFree(d_output_image);

    return h_output_image;
}

__global__ void demosaicKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads) {
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;

    if (thread_id >= total_threads) {
        return;
    }

    int y = thread_id / width;
    int x = thread_id % width;
    int idx = y * width + x;
    int R = 0, G = 0, B = 0;

    // Boundary check
    if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2) {
        output_image[3 * idx] = 0;
        output_image[3 * idx + 1] = 0;
        output_image[3 * idx + 2] = 0;
        return;
    }

    // Malvar interpolation for RGGB pattern
    if ((y % 2 == 0) && (x % 2 == 0)) {  // Red pixel
        R = input_image[idx];
        G = (4 * input_image[idx] - input_image[idx - 2 * width] - input_image[idx - 2] - input_image[idx + 2 * width] - input_image[idx + 2] +
            2 * (input_image[idx - width] + input_image[idx + width] + input_image[idx - 1] + input_image[idx + 1])) >> 3;
        B = (6 * input_image[idx] - ((3 * (input_image[idx - 2 * width] + input_image[idx - 2] + input_image[idx + 2 * width] + input_image[idx + 2])) >> 1) +
            2 * (input_image[idx - width - 1] + input_image[idx - width + 1] + input_image[idx + width - 1] + input_image[idx + width + 1])) >> 3;
    } else if ((y % 2 == 0) && (x % 2 == 1)) {  // Green pixel on Red row
        R = (5 * input_image[idx] - input_image[idx - 2] - input_image[idx - width - 1] - input_image[idx + width - 1] -
            input_image[idx - width + 1] - input_image[idx + width + 1] - input_image[idx + 2] +
            ((input_image[idx - 2 * width] + input_image[idx + 2 * width]) >> 1) + 4 * (input_image[idx - 1] + input_image[idx + 1])) >> 3;
        G = input_image[idx];
        B = (5 * input_image[idx] - input_image[idx - 2 * width] - input_image[idx - width - 1] - input_image[idx - width + 1] -
            input_image[idx + 2 * width] - input_image[idx + width - 1] - input_image[idx + width + 1] +
            ((input_image[idx - 2] + input_image[idx + 2]) >> 1) + 4 * (input_image[idx - width] + input_image[idx + width])) >> 3;
    } else if ((y % 2 == 1) && (x % 2 == 0)) {  // Green pixel on Blue row
        R = (5 * input_image[idx] - input_image[idx - 2 * width] - input_image[idx - width - 1] - input_image[idx - width + 1] -
            input_image[idx + 2 * width] - input_image[idx + width - 1] - input_image[idx + width + 1] +
            ((input_image[idx - 2] - input_image[idx + 2]) >> 1) + 4 * (input_image[idx - width] + input_image[idx + width])) >> 3;
        G = input_image[idx];
        B = (5 * input_image[idx] - input_image[idx - 2] - input_image[idx - width - 1] - input_image[idx + width - 1] -
            input_image[idx - width + 1] - input_image[idx + width + 1] - input_image[idx + 2] +
            ((input_image[idx - 2 * width] + input_image[idx + 2 * width]) >> 1) + 4 * (input_image[idx - 1] + input_image[idx + 1])) >> 3;
    } else {  // Blue pixel
        R = (6 * input_image[idx] - ((3 * (input_image[idx - 2 * width] + input_image[idx - 2] + input_image[idx + 2 * width] + input_image[idx + 2])) >> 1) +
            2 * (input_image[idx - width - 1] + input_image[idx - width + 1] + input_image[idx + width - 1] + input_image[idx + width + 1])) >> 3;
        G = (4 * input_image[idx] - input_image[idx - 2 * width] - input_image[idx - 2] - input_image[idx + 2 * width] - input_image[idx + 2] +
            2 * (input_image[idx - width] + input_image[idx + width] + input_image[idx - 1] + input_image[idx + 1])) >> 3;
        B = input_image[idx];
    }

    // Clip values to [0, 255]
    R = max(0, min(255, R));
    G = max(0, min(255, G));
    B = max(0, min(255, B));

    // Store the computed RGB values
    output_image[3 * idx] = R;
    output_image[3 * idx + 1] = G;
    output_image[3 * idx + 2] = B;
}

#endif

