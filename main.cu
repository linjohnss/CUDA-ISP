#include <stdio.h>
#include <string.h>
#include <dirent.h> 
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "image.h"
#include "filters/demosaic.h"
#include "filters/autoWhiteBalance.h"
#include "filters/sharpen_filter.h"
#include "filters/sharpness.h"
#include "filters/denoise.h"

const char* CPU_MODE = "cpu";
const char* GPU_MODE = "gpu";
const char* DEGRAD_MODE = "degrad";


// Define a function to calculate time difference
double calculate_time(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

void addnoise(uint8_t* rgb_buf, uint32_t width, uint32_t height) {
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            int idx = i * width + j;
            int noise = rand() % 10;
            for (int c = 0; c < 4; c++) {
                int value = rgb_buf[4 * idx + c] + noise;
                if (c == 1)
                    value+= 20;
                rgb_buf[4 * idx + c] = (uint8_t)(value > 255 ? 255 : (value < 0 ? 0 : value));
            }
        }
    }

}

void mosaicToRaw(uint8_t* image, uint8_t* raw_buf, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 4;  // Assume image has 4 channels (RGBA)
            int raw_idx = y * width + x;

            uint8_t R = image[idx];
            uint8_t G = image[idx + 1];
            uint8_t B = image[idx + 2];

            // RGGB pattern
            if (y % 2 == 0) {
                if (x % 2 == 0) {
                    raw_buf[raw_idx] = R;  // Red
                } else {
                    raw_buf[raw_idx] = G;  // Green (on red row)
                }
            } else {
                if (x % 2 == 0) {
                    raw_buf[raw_idx] = G;  // Green (on blue row)
                } else {
                    raw_buf[raw_idx] = B;  // Blue
                }
            }
        }
    }
}

int main(int argc, const char* argv[]) {
    if (argc != 4) {
        printf("Incorrect number of arguments.\n");
        return 1;
    }

    const char* path_to_input_image = argv[1];
    const char* path_to_output_image = argv[2];
    const char* mode = argv[3];

    int kernel_size = 3;

    if (strcmp(mode, CPU_MODE) == 0) {
        printf("CPU mode\n");
        int width, height, channels;
        uint8_t* raw_buf = loadImage(path_to_input_image, &width, &height, &channels, 1);
        if (raw_buf == NULL) {
            printf("Could not load image %s.\n", path_to_input_image);
            return 1;
        }
        printf("Processing image size: %d x %d\n", width, height);
        uint8_t* rgb_buf = (uint8_t*)malloc(sizeof(uint8_t) * width * height * 3);
        
        struct timespec global_start, global_end, start, end;

        // Start global timer
        clock_gettime(CLOCK_MONOTONIC, &global_start);

        // Measure demosaic time
        clock_gettime(CLOCK_MONOTONIC, &start);
        demosaic(raw_buf, rgb_buf, width, height);
        clock_gettime(CLOCK_MONOTONIC, &end);
        printf("Demosaic time: %.6f seconds\n", calculate_time(start, end));

        // Measure autoWhiteBalance time
        clock_gettime(CLOCK_MONOTONIC, &start);
        autoWhiteBalance(rgb_buf, width, height);
        clock_gettime(CLOCK_MONOTONIC, &end);
        printf("Auto White Balance time: %.6f seconds\n", calculate_time(start, end));

        // Measure denoise time
        clock_gettime(CLOCK_MONOTONIC, &start);
        denoise(rgb_buf, width, height, kernel_size);
        clock_gettime(CLOCK_MONOTONIC, &end);
        printf("Denoise time: %.6f seconds\n", calculate_time(start, end));

        // Measure sharpness time
        clock_gettime(CLOCK_MONOTONIC, &start);
        sharpness(rgb_buf, width, height);
        clock_gettime(CLOCK_MONOTONIC, &end);
        printf("Sharpness time: %.6f seconds\n", calculate_time(start, end));

        // End global timer
        clock_gettime(CLOCK_MONOTONIC, &global_end);
        printf("Total processing time: %.6f seconds\n", calculate_time(global_start, global_end));

        printf("Finish\n");
        writeImage(path_to_output_image, (stbi_uc*)rgb_buf, width, height, 3);
        
        // Free allocated memory
        free(raw_buf);
        free(rgb_buf);
    } else if (strcmp(mode, GPU_MODE) == 0) {
        printf("GPU mode\n");
        int width, height, channels;
        uint8_t* raw_buf = loadImage(path_to_input_image, &width, &height, &channels, 1);
        if (raw_buf == NULL) {
            printf("Could not load image %s.\n", path_to_input_image);
            return 1;
        }
        printf("Processing image size: %d x %d\n", width, height);

        struct timespec global_start, global_end, start, end;
        // Start global timer
        clock_gettime(CLOCK_MONOTONIC, &global_start);

        stbi_uc* filtered_image;

        clock_gettime(CLOCK_MONOTONIC, &start);
        filtered_image = demosaic((stbi_uc*)raw_buf, width, height, 3);
        clock_gettime(CLOCK_MONOTONIC, &end);
        printf("Demosaic time: %.6f seconds\n", calculate_time(start, end));

        clock_gettime(CLOCK_MONOTONIC, &start);
        filtered_image = autoWhiteBalance(filtered_image, width, height, 3);
        clock_gettime(CLOCK_MONOTONIC, &end);
        printf("Auto White Balance time: %.6f seconds\n", calculate_time(start, end));

        clock_gettime(CLOCK_MONOTONIC, &start);
        filtered_image = denoise(filtered_image, width, height, 3, kernel_size);
        clock_gettime(CLOCK_MONOTONIC, &end);
        printf("Denoise time: %.6f seconds\n", calculate_time(start, end));

        clock_gettime(CLOCK_MONOTONIC, &start);
        filtered_image = sharpness(filtered_image, width, height, 3);
        clock_gettime(CLOCK_MONOTONIC, &end);
        printf("Sharpness time: %.6f seconds\n", calculate_time(start, end));

        // End global timer
        clock_gettime(CLOCK_MONOTONIC, &global_end);
        printf("Total processing time: %.6f seconds\n", calculate_time(global_start, global_end));

        printf("Finish\n");
        writeImage(path_to_output_image, filtered_image, width, height, 3);
        
        // Free allocated memory
        free(raw_buf);
    } else if (strcmp(mode, DEGRAD_MODE) == 0) {
        int width, height, channels;
        stbi_uc* image = loadImage(path_to_input_image, &width, &height, &channels, 4);

        // mosaic image to raw data (RGGB pattern) image contains 4 channels
        uint8_t* raw_buf = (uint8_t*)malloc(sizeof(uint8_t) * width * height);

        addnoise(image, width, height);

        mosaicToRaw(image, raw_buf, width, height);

        writeImage(path_to_output_image, (stbi_uc*)raw_buf, width, height, 1);
        
    }
    return 0;
}