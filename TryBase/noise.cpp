#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>
using namespace std;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Apply Mean Filter (smoothing)
void applyMeanFilter(unsigned char *img, unsigned char *output, int width, int height, int channels, int filterSize)
{
    int offset = filterSize / 2;

#pragma omp parallel for collapse(2)
    for (int y = offset; y < height - offset; y++)
    {
        for (int x = offset; x < width - offset; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                int sum = 0;
                for (int ky = -offset; ky <= offset; ky++)
                {
                    for (int kx = -offset; kx <= offset; kx++)
                    {
                        sum += img[((y + ky) * width + (x + kx)) * channels + c];
                    }
                }
                int avg = sum / (filterSize * filterSize);
                output[(y * width + x) * channels + c] = avg;
            }
        }
    }
}

// Apply Median Filter
void applyMedianFilter(unsigned char *img, unsigned char *output, int width, int height, int channels, int filterSize)
{
    int offset = filterSize / 2;

#pragma omp parallel for collapse(2)
    for (int y = offset; y < height - offset; y++)
    {
        for (int x = offset; x < width - offset; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                vector<int> neighborhood;
                // Collect the neighborhood values
                for (int ky = -offset; ky <= offset; ky++)
                {
                    for (int kx = -offset; kx <= offset; kx++)
                    {
                        neighborhood.push_back(img[((y + ky) * width + (x + kx)) * channels + c]);
                    }
                }
                // Sort and find the median
                sort(neighborhood.begin(), neighborhood.end());
                int median = neighborhood[neighborhood.size() / 2];
                output[(y * width + x) * channels + c] = median;
            }
        }
    }
}

// Combine both filters by averaging the output images
void combineFilters(unsigned char *meanFiltered, unsigned char *medianFiltered, unsigned char *combined, int width, int height, int channels)
{
#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                // Combine by averaging the pixel values
                int meanValue = meanFiltered[(y * width + x) * channels + c];
                int medianValue = medianFiltered[(y * width + x) * channels + c];
                combined[(y * width + x) * channels + c] = (meanValue + medianValue) / 2;
            }
        }
    }
}

int main()
{
    int width, height, channels;
    unsigned char *img = stbi_load("noise_input.jpg", &width, &height, &channels, 0);

    if (img == NULL)
    {
        cout << "Error loading image\n";
        return -1;
    }

    // Create output image buffers for processing
    unsigned char *imgMeanFiltered = new unsigned char[width * height * channels];
    unsigned char *imgMedianFiltered = new unsigned char[width * height * channels];
    unsigned char *imgCombined = new unsigned char[width * height * channels];

    // Apply Mean Filter (with a 3x3 filter size)
    applyMeanFilter(img, imgMeanFiltered, width, height, channels, 3);

    // Apply Median Filter (with a 3x3 filter size)
    applyMedianFilter(img, imgMedianFiltered, width, height, channels, 3);

    // Combine the Mean and Median Filter results
    combineFilters(imgMeanFiltered, imgMedianFiltered, imgCombined, width, height, channels);

    // Save the combined filtered image
    stbi_write_jpg("output_noise.jpg", width, height, channels, imgCombined, 100);

    // Free the image memory
    stbi_image_free(img);
    delete[] imgMeanFiltered;
    delete[] imgMedianFiltered;
    delete[] imgCombined;

    return 0;
}
