#ifndef SHARPNESS_H
#define SHARPNESS_H

#include "../image.h"
#include "util.h"
#include "edge_detection_filter.h"

stbi_uc* sharpness(stbi_uc* input_image, int width, int height, int channels);
__global__ void alphaBlendKernel(stbi_uc* input_image, stbi_uc* edge_image, stbi_uc* output_image, int width, int height, int channels, int total_threads);

stbi_uc* sharpness(stbi_uc* input_image, int width, int height, int channels) {
    Memory memory = Global;
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

    red = current_pixel.r + edge_pixel.r;
    green = current_pixel.g + edge_pixel.g;
    blue = current_pixel.b + edge_pixel.b;

    Pixel pixel;
    if (red < 0) {
        pixel.r = 0;
    } else if (red > 255) {
        pixel.r = 255;
    } else {
        pixel.r = red;
    }
    if (green < 0) {
        pixel.g = 0;
    } else if (green > 255) {
        pixel.g = 255;
    } else {
        pixel.g = green;
    }
    if (blue < 0) {
        pixel.b = 0;
    } else if (blue > 255) {
        pixel.b = 255;
    } else {
        pixel.b = blue;
    }

    setPixel(output_image, width, x, y, &pixel);
}

#endif