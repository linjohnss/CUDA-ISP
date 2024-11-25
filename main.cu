#include <stdio.h>
#include <string.h>
#include <dirent.h> 
#include <string.h>
#include <stdlib.h>

#include "image.h"
#include "filters/blur_filter.h"
#include "filters/demosaic.h"
#include "filters/autoWhiteBalance.h"
#include "filters/sharpen_filter.h"
#include "filters/edge_detection_filter.h"
#include "filters/sharpness.h"

const char* BLUR_FILTER = "blur";
const char* SHARPEN_FILTER = "sharpen";
const char* EDGE_DETECTION_FILTER = "edge";

const char* CPU_MODE = "cpu";
const char* GPU_MODE = "gpu";
const char* BATCH_MODE = "batch";
const char* DEGRAD_MODE = "degrad";

// void demosaic(uint8_t* raw_buf, uint8_t* rgb_buf, uint32_t width, uint32_t height) {
//     for (uint32_t i = 1; i < height - 1; i++) {
//         for (uint32_t j = 1; j < width - 1; j++) {
//             int idx = i * width + j;
//             uint8_t R = 0, G = 0, B = 0;

//             // RGGB pattern: Assign R, G, B based on pixel location in the Bayer pattern
//             if ((i % 2 == 0) && (j % 2 == 0)) {  // Red pixel
//                 R = raw_buf[idx];
//                 G = (raw_buf[idx - 1] + raw_buf[idx + 1] + raw_buf[idx - width] + raw_buf[idx + width]) >> 2;
//                 B = (raw_buf[idx - width - 1] + raw_buf[idx - width + 1] + raw_buf[idx + width - 1] + raw_buf[idx + width + 1]) >> 2;
//             } else if ((i % 2 == 0) && (j % 2 == 1)) {  // Green pixel on Red row
//                 R = (raw_buf[idx - 1] + raw_buf[idx + 1]) >> 1;
//                 G = raw_buf[idx];
//                 B = (raw_buf[idx - width] + raw_buf[idx + width]) >> 1;
//             } else if ((i % 2 == 1) && (j % 2 == 0)) {  // Green pixel on Blue row
//                 R = (raw_buf[idx - width] + raw_buf[idx + width]) >> 1;
//                 G = raw_buf[idx];
//                 B = (raw_buf[idx - 1] + raw_buf[idx + 1]) >> 1;
//             } else {  // Blue pixel
//                 R = (raw_buf[idx - width - 1] + raw_buf[idx - width + 1] + raw_buf[idx + width - 1] + raw_buf[idx + width + 1]) >> 2;
//                 G = (raw_buf[idx - 1] + raw_buf[idx + 1] + raw_buf[idx - width] + raw_buf[idx + width]) >> 2;
//                 B = raw_buf[idx];
//             }

//             // Store RGB values
//             rgb_buf[3 * idx] = R;
//             rgb_buf[3 * idx + 1] = G;
//             rgb_buf[3 * idx + 2] = B;
//         }
//     }
// }

void demosaic(uint8_t* raw_buf, uint8_t* rgb_buf, uint32_t width, uint32_t height) {
    for (uint32_t i = 2; i < height - 2; i++) {
        for (uint32_t j = 2; j < width - 2; j++) {
            int idx = i * width + j;
            int R = 0, G = 0, B = 0;

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
            R = std::max(0, std::min(255, R));
            G = std::max(0, std::min(255, G));
            B = std::max(0, std::min(255, B));

            // Store RGB values
            rgb_buf[3 * idx] = static_cast<uint8_t>(R);
            rgb_buf[3 * idx + 1] = static_cast<uint8_t>(G);
            rgb_buf[3 * idx + 2] = static_cast<uint8_t>(B);
        }
    }
}


void denoise(uint8_t* rgb_buf, uint32_t width, uint32_t height) {
    for (uint32_t i = 1; i < height - 1; i++) {
        for (uint32_t j = 1; j < width - 1; j++) {
            int idx = i * width + j;
            uint8_t R = 0, G = 0, B = 0;

            // Apply 3x3 median filter
            uint8_t R_values[9] = {rgb_buf[3 * (idx - width - 1)], rgb_buf[3 * (idx - width)], rgb_buf[3 * (idx - width + 1)],
                                   rgb_buf[3 * (idx - 1)], rgb_buf[3 * idx], rgb_buf[3 * (idx + 1)],
                                   rgb_buf[3 * (idx + width - 1)], rgb_buf[3 * (idx + width)], rgb_buf[3 * (idx + width + 1)]};
            uint8_t G_values[9] = {rgb_buf[3 * (idx - width - 1) + 1], rgb_buf[3 * (idx - width) + 1], rgb_buf[3 * (idx - width + 1) + 1],
                                   rgb_buf[3 * (idx - 1) + 1], rgb_buf[3 * idx + 1], rgb_buf[3 * (idx + 1) + 1],
                                   rgb_buf[3 * (idx + width - 1) + 1], rgb_buf[3 * (idx + width) + 1], rgb_buf[3 * (idx + width + 1) + 1]};
            uint8_t B_values[9] = {rgb_buf[3 * (idx - width - 1) + 2], rgb_buf[3 * (idx - width) + 2], rgb_buf[3 * (idx - width + 1) + 2],
                                   rgb_buf[3 * (idx - 1) + 2], rgb_buf[3 * idx + 2], rgb_buf[3 * (idx + 1) + 2],
                                   rgb_buf[3 * (idx + width - 1) + 2], rgb_buf[3 * (idx + width) + 2], rgb_buf[3 * (idx + width + 1) + 2]};

            // Sort RGB values
            for (int k = 0; k < 9; k++) {
                for (int l = k + 1; l < 9; l++) {
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
            R = R_values[4];
            G = G_values[4];
            B = B_values[4];

            // Store RGB values
            rgb_buf[3 * idx] = R;
            rgb_buf[3 * idx + 1] = G;
            rgb_buf[3 * idx + 2] = B;
        }
    }
}

#define GAIN_FRACTION_BITS 6 //awb gain control
#define clip_max(x, value) x > value ? value : x
void autoWhiteBalanceRaw(uint16_t* raw_buf, uint32_t width, uint32_t height) {
    uint16_t max_value = (1 << 10) - 1;;
    uint32_t idx;
    uint32_t tmp;
    uint8_t r_gain, gr_gain, gb_gain, b_gain;
    r_gain = (uint8_t)(1.5 * (1 << GAIN_FRACTION_BITS));
    gr_gain = (uint8_t)(1.0 * (1 << GAIN_FRACTION_BITS));
    gb_gain = (uint8_t)(1.0 * (1 << GAIN_FRACTION_BITS));
    b_gain = (uint8_t)(1.1 * (1 << GAIN_FRACTION_BITS));

    for (uint16_t i = 0; i < height; i += 2)
    {
        for (uint16_t j = 0; j < width; j += 2)
        {
            idx = i * width + j;
            tmp = (r_gain * raw_buf[idx]) >> GAIN_FRACTION_BITS;
            raw_buf[idx] = clip_max(tmp, max_value);
            idx += 1;
            tmp = (gr_gain * raw_buf[idx]) >> GAIN_FRACTION_BITS;
            raw_buf[idx] = clip_max(tmp, max_value);
            idx += width;
            tmp = (b_gain * raw_buf[idx]) >> GAIN_FRACTION_BITS;
            raw_buf[idx] = clip_max(tmp, max_value);
            idx -= 1;
            tmp = (gb_gain * raw_buf[idx]) >> GAIN_FRACTION_BITS;
            raw_buf[idx] = clip_max(tmp, max_value);
        }
    }
}

uint8_t clamp_cpu(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return (uint8_t)value;
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

    if (strcmp(mode, CPU_MODE) == 0) {
        int width, height, channels;
        uint8_t* raw_buf = loadImage(path_to_input_image, &width, &height, &channels, 1);
        if (raw_buf == NULL) {
            printf("Could not load image %s.\n", path_to_input_image);
            return 1;
        }
        printf("width: %d, height: %d, channels: %d\n", width, height, channels);
        uint8_t* rgb_buf = (uint8_t*)malloc(sizeof(uint8_t) * width * height * 3);
        
        demosaic(raw_buf, rgb_buf, width, height);

        denoise(rgb_buf, width, height);

        autoWhiteBalance(rgb_buf, width, height);

        printf("Finish\n");
        writeImage(path_to_output_image, (stbi_uc*)rgb_buf, width, height, 3);
        
        // Free allocated memory
        free(raw_buf);
        free(rgb_buf);
    } else if (strcmp(mode, GPU_MODE) == 0) {
        int width, height, channels;
        uint8_t* raw_buf = loadImage(path_to_input_image, &width, &height, &channels, 1);
        if (raw_buf == NULL) {
            printf("Could not load image %s.\n", path_to_input_image);
            return 1;
        }

        stbi_uc* filtered_image;
        filtered_image = demosaic((stbi_uc*)raw_buf, width, height, 3);

        filtered_image = blur(filtered_image, width, height, 3);

        filtered_image = sharpness(filtered_image, width, height, 3);

        filtered_image = autoWhiteBalance(filtered_image, width, height, 3);
        
        printf("Finish\n");
        writeImage(path_to_output_image, filtered_image, width, height, 3);
        
        // Free allocated memory
        free(raw_buf);
    } else if (strcmp(mode, DEGRAD_MODE) == 0) {
        int width, height, channels;
        stbi_uc* image = loadImage(path_to_input_image, &width, &height, &channels, 4);

        // mosaic image to raw data (RGGB pattern) image contains 4 channels
        uint8_t* raw_buf = (uint8_t*)malloc(sizeof(uint8_t) * width * height);

        mosaicToRaw(image, raw_buf, width, height);

        writeImage(path_to_output_image, (stbi_uc*)raw_buf, width, height, 1);
        
    }
    return 0;
}