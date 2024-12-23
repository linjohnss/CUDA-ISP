#ifndef EDGE_DETECTION_FILTER_H
#define EDGE_DETECTION_FILTER_H

#include "../image.h"
#include "convolve.h"

int edge_mask_x_3[] = {-1, 0, 1,
                -2, 0, 2,
                -1, 0, 1};
int edge_mask_y_3[] = {1,  2,  1,
                0,  0,  0,
               -1, -2, -1};

int edge_mask_dimension_3 = 3;

int mask_x_5[] = {-2, -1, 0, 1, 2,
                -2, -1, 0, 2, 2,
                -4, -2, 0, 2, 4,
                -2, -1, 0, 1, 2,
                -2, -1, 0, 1, 2};
int mask_y_5[] = {2,  2,  4,  2,  2,
                1,  1,  2,  1,  1,
                0,  0,  0,  0,  0,
                -1, -1, -2, -1, -1,
                -2, -2, -4, -2, -2};

int mask_dimension_5 = 5;

int mask_x_7[] = {-3, -2, -1, 0, 1, 2, 3,
                -3, -2, -1, 0, 2, 2, 3,
                -3, -2, -1, 0, 2, 2, 3,
                -6, -4, -2, 0, 2, 4, 6,
                -3, -2, -1, 0, 1, 2, 3,
                -3, -2, -1, 0, 1, 2, 3,
                -3, -2, -1, 0, 1, 2, 3};
int mask_y_7[] = {3,  3,  3,  6,  3,  3,  3,
                2,  2,  2,  4,  2,  2,  2,
                1,  1,  1,  2,  1,  1,  1,
                0,  0,  0,  0,  0,  0,  0,
               -1, -1, -1, -2, -1, -1, -1,
               -2, -2, -2, -4, -2, -2, -1,
               -3, -3, -3, -6, -3, -3, -3};

int mask_dimension_7 = 7;

stbi_uc* edgeDetection(stbi_uc* input_image, int width, int height, int channels) {
    stbi_uc* output = convolve(input_image, width, height, channels, edge_mask_x_3, edge_mask_dimension_3);
    output = convolve(output, width, height, channels, edge_mask_y_3, edge_mask_dimension_3);

    return output;
}

#endif