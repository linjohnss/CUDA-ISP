#ifndef MEDIAN_FILTER_H
#define MEDIAN_FILTER_H

#include "../image.h"
#include "util.h"

stbi_uc* denoise(stbi_uc* input_image, int width, int height, int channels, int mask_size);
__global__ void denoiseKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int mask_size);
void denoise(uint8_t* rgb_buf, uint32_t width, uint32_t height, int kernel_size);

stbi_uc* denoise(stbi_uc* input_image, int width, int height, int channels, int mask_size) {
    int image_size = channels * width * height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_output_image;
    stbi_uc* h_output_image = (stbi_uc*)malloc(image_size);

    cudaMallocManaged(&d_input_image, image_size);
    cudaMallocManaged(&d_output_image, image_size);
    cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    denoiseKernel<<<blocks, threads>>>(d_input_image, d_output_image, width, height, channels, mask_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;
}

__global__ void denoiseKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int mask_size) {
    int y_coordinate = blockIdx.y * blockDim.y + threadIdx.y;
    int x_coordinate = blockIdx.x * blockDim.x + threadIdx.x;

    if (x_coordinate < mask_size / 2 || x_coordinate >= width - mask_size / 2 || y_coordinate < mask_size / 2 || y_coordinate >= height - mask_size / 2) {
        return;
    }

    Pixel current_pixel;
    int mask_area = mask_size * mask_size;
    int radius = mask_size / 2;
    
    int r_values[100], g_values[100], b_values[100];
    int count = 0;

    for (int i = 0; i < mask_size; i++) {
        for (int j = 0; j < mask_size; j++) {
            getPixel(input_image, width, x_coordinate - radius + i, y_coordinate - radius + j, &current_pixel);
            r_values[count] = current_pixel.r;
            g_values[count] = current_pixel.g;
            b_values[count] = current_pixel.b;
            count++;
        }
    }

    // Sort arrays to find median
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (r_values[i] > r_values[j]) {
                int temp = r_values[i];
                r_values[i] = r_values[j];
                r_values[j] = temp;
            }
            if (g_values[i] > g_values[j]) {
                int temp = g_values[i];
                g_values[i] = g_values[j];
                g_values[j] = temp;
            }
            if (b_values[i] > b_values[j]) {
                int temp = b_values[i];
                b_values[i] = b_values[j];
                b_values[j] = temp;
            }
        }
    }

    Pixel pixel;
    pixel.r = r_values[count / 2];
    pixel.g = g_values[count / 2];
    pixel.b = b_values[count / 2];

    setPixel(output_image, width, x_coordinate, y_coordinate, &pixel);
}

void denoise(uint8_t* rgb_buf, uint32_t width, uint32_t height, int kernel_size) {
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            if (i < kernel_size / 2 || i >= height - kernel_size / 2 || j < kernel_size / 2 || j >= width - kernel_size / 2) {
                continue;
            }
            int idx = i * width + j;
            uint8_t R = 0, G = 0, B = 0;
            // int kernel_size = 3;
            int kernel_area = kernel_size * kernel_size;
            int padding_size = kernel_size / 2;

            uint8_t R_values[kernel_area], G_values[kernel_area], B_values[kernel_area];

            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    int pixel_idx = ((i + k - padding_size) * width + (j + l - padding_size)) * 3;
                    R_values[k * kernel_size + l] = rgb_buf[pixel_idx];
                    G_values[k * kernel_size + l] = rgb_buf[pixel_idx + 1];
                    B_values[k * kernel_size + l] = rgb_buf[pixel_idx + 2];
                }
            }
            
            // Sort RGB values
            for (int k = 0; k < kernel_size; k++) {
                for (int l = k + 1; l < kernel_size; l++) {
                    if (R_values[k] > R_values[l]) {
                        uint8_t temp = R_values[k];
                        R_values[k] = R_values[l];
                        R_values[l] = temp;
                    }
                    if (G_values[k] > G_values[l]) {
                        uint8_t temp = G_values[k];
                        G_values[k] = G_values[l];
                        G_values[l] = temp;
                    }
                    if (B_values[k] > B_values[l]) {
                        uint8_t temp = B_values[k];
                        B_values[k] = B_values[l];
                        B_values[l] = temp;
                    }
                }
            }

            // Assign median RGB values
            R = R_values[kernel_area / 2];
            G = G_values[kernel_area / 2];
            B = B_values[kernel_area / 2];

            // Store RGB values
            rgb_buf[3 * idx] = R;
            rgb_buf[3 * idx + 1] = G;
            rgb_buf[3 * idx + 2] = B;
        }
    }
}

#endif
