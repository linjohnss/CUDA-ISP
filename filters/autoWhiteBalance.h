#ifndef AUTOWHITEBALANCE_H
#define AUTOWHITEBALANCE_H

#include "../image.h"
#include "util.h"

stbi_uc* autoWhiteBalance(stbi_uc* input_image, int width, int height, int channels);
__global__ void autoWhiteBalanceSumKernel(stbi_uc* input_image, int width, int height, int channels, unsigned long long* sum_R, unsigned long long* sum_G, unsigned long long* sum_B);
__global__ void autoWhiteBalanceApplyKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, float r_gain, float b_gain);
void autoWhiteBalance(uint8_t* rgb_buf, uint32_t width, uint32_t height);

stbi_uc* autoWhiteBalance(stbi_uc* input_image, int width, int height, int channels) {
    int image_size = channels * width * height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_output_image;
    stbi_uc* h_output_image = (stbi_uc*) malloc(image_size);
    
    cudaMallocManaged(&d_input_image, image_size);
    cudaMallocManaged(&d_output_image, image_size);
    cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice);

    // Allocate memory for RGB sums
    unsigned long long *d_sum_R, *d_sum_G, *d_sum_B;
    cudaMallocManaged(&d_sum_R, sizeof(unsigned long long));
    cudaMallocManaged(&d_sum_G, sizeof(unsigned long long));
    cudaMallocManaged(&d_sum_B, sizeof(unsigned long long));

    // Initialize sums to 0
    *d_sum_R = 0;
    *d_sum_G = 0;
    *d_sum_B = 0;

    int total_threads = width * height;
    int threads = min(THREADS_PER_BLOCK, total_threads);
    int blocks = (total_threads + threads - 1) / threads;
    
    // First kernel to calculate the sum of RGB values
    autoWhiteBalanceSumKernel<<<blocks, threads>>>(d_input_image, width, height, channels, d_sum_R, d_sum_G, d_sum_B);
    cudaDeviceSynchronize();

    // Calculate average RGB values and gains
    int pixel_count = width * height;
    float avg_R = *d_sum_R / (float)pixel_count;
    float avg_G = *d_sum_G / (float)pixel_count;
    float avg_B = *d_sum_B / (float)pixel_count;

    float r_gain = avg_G / avg_R;
    float b_gain = avg_G / avg_B;

    // Second kernel to apply the white balance gains
    autoWhiteBalanceApplyKernel<<<blocks, threads>>>(d_input_image, d_output_image, width, height, channels, r_gain, b_gain);
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(h_output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input_image);
    cudaFree(d_output_image);
    cudaFree(d_sum_R);
    cudaFree(d_sum_G);
    cudaFree(d_sum_B);

    return h_output_image;
}

__global__ void autoWhiteBalanceSumKernel(stbi_uc* input_image, int width, int height, int channels, unsigned long long* sum_R, unsigned long long* sum_G, unsigned long long* sum_B) {
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int image_size = width * height;
    int idx = thread_id * channels;

    if (thread_id < image_size) {
        atomicAdd(sum_R, (unsigned long long)input_image[idx]);
        atomicAdd(sum_G, (unsigned long long)input_image[idx + 1]);
        atomicAdd(sum_B, (unsigned long long)input_image[idx + 2]);
    }
}

__global__ void autoWhiteBalanceApplyKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, float r_gain, float b_gain) {
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int image_size = width * height;
    int idx = thread_id * channels;

    if (thread_id < image_size) {
        output_image[idx] = (stbi_uc)clamp((input_image[idx] * r_gain), 0.0f, 255.0f);
        output_image[idx + 1] = (stbi_uc)clamp((input_image[idx + 1]), 0.0f, 255.0f);
        output_image[idx + 2] = (stbi_uc)clamp((input_image[idx + 2] * b_gain), 0.0f, 255.0f);
    }
}

void autoWhiteBalance(uint8_t* rgb_buf, uint32_t width, uint32_t height) {
    uint32_t image_size = width * height;
    uint64_t r_sum = 0, g_sum = 0, b_sum = 0;

    // Calculate the sum of each channel
    for (uint32_t i = 0; i < image_size; i++) {
        r_sum += rgb_buf[3 * i];
        g_sum += rgb_buf[3 * i + 1];
        b_sum += rgb_buf[3 * i + 2];
    }

    // Compute average intensity for each color channel
    float r_avg = r_sum / (float)image_size;
    float g_avg = g_sum / (float)image_size;
    float b_avg = b_sum / (float)image_size;

    // Calculate gain for each channel to match the green channel
    float r_gain = g_avg / r_avg;
    float b_gain = g_avg / b_avg;

    // Apply the white balance gains
    for (uint32_t i = 0; i < image_size; i++) {
        int r = rgb_buf[3 * i];
        int g = rgb_buf[3 * i + 1];
        int b = rgb_buf[3 * i + 2];

        rgb_buf[3 * i] = (uint8_t)clamp_cpu(r * r_gain, 0, 255);
        rgb_buf[3 * i + 1] = (uint8_t)clamp_cpu(g, 0, 255);         // No adjustment for green
        rgb_buf[3 * i + 2] = (uint8_t)clamp_cpu(b * b_gain, 0, 255);
    }
}

#endif
