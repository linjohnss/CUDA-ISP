#ifndef CONVOLVE_H
#define CONVOLVE_H

#include "../image.h"
#include "util.h"

stbi_uc* convolve(const stbi_uc* input_image, int width, int height, int channels, const int* mask, int mask_dimension);
__global__ void convolve(const stbi_uc* input_image, stbi_uc* output_image, const int width, const int height, const int* mask, const int mask_dimension);
void convolve(uint8_t* rgb_buf, uint8_t* edge_buf, uint32_t width, uint32_t height, const int* mask_x, const int* mask_y);

stbi_uc* convolve(const stbi_uc* input_image, int width, int height, int channels, const int* mask, int mask_dimension) {
    int* d_mask;

    cudaMallocManaged(&d_mask, mask_dimension * mask_dimension * sizeof(int));
    cudaMemcpy(d_mask, mask, mask_dimension * mask_dimension * sizeof(int), cudaMemcpyHostToDevice);
    
    int image_size = channels * width * height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_output_image;
    stbi_uc* h_output_image = (stbi_uc*) malloc(image_size);

    for (int i = 0; i < width * height; i++) {
        h_output_image[i] = input_image[i];
    }

    cudaMallocManaged(&d_input_image, image_size);
    cudaMallocManaged(&d_output_image, image_size);
    cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice);

    int total_threads = width * height;
    int threads = min(MAX_THREADS, total_threads);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    convolve<<<grid, block>>>(d_input_image, d_output_image, width, height, d_mask, mask_dimension);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;
}

__global__ void convolve(const stbi_uc* input_image, stbi_uc* output_image, const int width, const int height, const int* mask, const int mask_dimension) {
    const int x_coordinate = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_coordinate = blockIdx.y * blockDim.y + threadIdx.y;
    
    int half_mask_size = mask_dimension / 2;

    Pixel current_pixel;
    int red = 0;
    int blue = 0;
    int green = 0;
    // int alpha = 0;
    for (int i = 0; i < mask_dimension; i++) {
        for (int j = 0; j < mask_dimension; j++) {
            int current_x_global = x_coordinate - half_mask_size + i;
            int current_y_global = y_coordinate - half_mask_size + j;
            if (isOutOfBounds(current_x_global, current_y_global, width, height)) {
                continue;
            }
            getPixel(input_image, width, current_x_global, current_y_global, &current_pixel);
            int mask_element = mask[i * mask_dimension + j];

            red += current_pixel.r * mask_element;
            green += current_pixel.g * mask_element;
            blue += current_pixel.b * mask_element;
        }
    }

    Pixel pixel;
    pixel.r = clamp(red, 0, 255);
    pixel.g = clamp(green, 0, 255);
    pixel.b = clamp(blue, 0, 255);

    setPixel(output_image, width, x_coordinate, y_coordinate, &pixel);
}

#endif