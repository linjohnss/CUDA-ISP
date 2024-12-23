#ifndef SHARPNESS_H
#define SHARPNESS_H

#include "../image.h"
#include "util.h"
#include "edge_detection_filter.h"

stbi_uc* sharpness(stbi_uc* input_image, int width, int height, int channels);
__global__ void alphaBlendKernel(stbi_uc* input_image, stbi_uc* edge_image, stbi_uc* output_image, int width, int height, int channels, int total_threads);
void sharpness(uint8_t* rgb_buf, uint32_t width, uint32_t height);

stbi_uc* sharpness(stbi_uc* input_image, int width, int height, int channels) {
    stbi_uc* edge_image = edgeDetection(input_image, width, height, channels);
    
    int image_size = channels * width * height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_edge_image;
    stbi_uc* d_output_image;
    stbi_uc* h_output_image = (stbi_uc*) malloc(image_size);
    for (int i = 0; i < width * height; i++) {
        h_output_image[i] = input_image[i];
    }

    cudaMallocManaged(&d_input_image, image_size);
    cudaMallocManaged(&d_edge_image, image_size);
    cudaMallocManaged(&d_output_image, image_size);
    cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_image, edge_image, image_size, cudaMemcpyHostToDevice);

    int total_threads = width * height;
    int threads = min(THREADS_PER_BLOCK, total_threads);
    int blocks = (threads == total_threads) ? 1 : total_threads / THREADS_PER_BLOCK;

    alphaBlendKernel<<<blocks, threads>>>(d_input_image, d_edge_image, d_output_image, width, height, channels, total_threads);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;
}

__global__ void alphaBlendKernel(stbi_uc* input_image, stbi_uc* edge_image, stbi_uc* output_image, int width, int height, int channels, int total_threads) {
    
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (thread_id >= total_threads) {
        return;
    }

    int x = thread_id % width;
    int y = thread_id / width;
    
    double red = 0;
    double blue = 0;
    double green = 0;

    Pixel current_pixel;
    getPixel(input_image, width, x, y, &current_pixel);
    Pixel edge_pixel;
    getPixel(edge_image, width, x, y, &edge_pixel);

    float alpha = 0.1;
    red = current_pixel.r + edge_pixel.r * alpha;
    green = current_pixel.g + edge_pixel.g * alpha;
    blue = current_pixel.b + edge_pixel.b * alpha;

    Pixel pixel;
    pixel.r = clamp(red, 0, 255);
    pixel.g = clamp(green, 0, 255);
    pixel.b = clamp(blue, 0, 255);
    
    setPixel(output_image, width, x, y, &pixel);
}

void sharpness(uint8_t* rgb_buf, uint32_t width, uint32_t height) {
    uint8_t* edge_buf = (uint8_t*)malloc(width * height * sizeof(uint8_t) * 3);
    if (!edge_buf) {
        printf("Memory allocation failed for edge buffer.\n");
        return;
    }

    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            int idx = i * width + j;
            int R = 0, G = 0, B = 0;
            if (i == 0 || i == height - 1 || j == 0 || j == width - 1) {
                edge_buf[idx * 3] = 0;
                edge_buf[idx * 3 + 1] = 0;
                edge_buf[idx * 3 + 2] = 0;
                continue;
            }

            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    int pixel_idx = ((i + k) * width + (j + l)) * 3;
                    R += rgb_buf[pixel_idx] * edge_mask_x_3[(k + 1) * 3 + (l + 1)];
                    G += rgb_buf[pixel_idx + 1] * edge_mask_x_3[(k + 1) * 3 + (l + 1)];
                    B += rgb_buf[pixel_idx + 2] * edge_mask_x_3[(k + 1) * 3 + (l + 1)];

                    
                }
            }
            edge_buf[idx * 3] = clamp_cpu(R, 0, 255);
            edge_buf[idx * 3 + 1] = clamp_cpu(G, 0, 255);
            edge_buf[idx * 3 + 2] = clamp_cpu(B, 0, 255);
        }
    }

    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            int idx = i * width + j;
            int R = 0, G = 0, B = 0;
            if (i == 0 || i == height - 1 || j == 0 || j == width - 1) {
                edge_buf[idx * 3] = 0;
                edge_buf[idx * 3 + 1] = 0;
                edge_buf[idx * 3 + 2] = 0;
                continue;
            }

            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    int pixel_idx = ((i + k) * width + (j + l)) * 3;
                    R += rgb_buf[pixel_idx] * edge_mask_y_3[(k + 1) * 3 + (l + 1)];
                    G += rgb_buf[pixel_idx + 1] * edge_mask_y_3[(k + 1) * 3 + (l + 1)];
                    B += rgb_buf[pixel_idx + 2] * edge_mask_y_3[(k + 1) * 3 + (l + 1)];
                }
            }
            edge_buf[idx * 3] = clamp_cpu(R, 0, 255);
            edge_buf[idx * 3 + 1] = clamp_cpu(G, 0, 255);
            edge_buf[idx * 3 + 2] = clamp_cpu(B, 0, 255);
        }
    }

    float alpha = 0.1;

    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            int idx = (i * width + j) * 3;
            uint8_t edge_value = edge_buf[i * width + j];

            for (int c = 0; c < 3; c++) {
                int enhanced_value = rgb_buf[idx + c] + alpha * edge_value;
                rgb_buf[idx + c] = (uint8_t)(enhanced_value > 255 ? 255 : (enhanced_value < 0 ? 0 : enhanced_value));
            }
        }
    }

    free(edge_buf);
}

#endif