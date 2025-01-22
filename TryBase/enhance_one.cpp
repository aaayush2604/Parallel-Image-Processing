#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
using namespace std;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Brightness Correction
void applyBrightnessCorrection(unsigned char *img, int width, int height, int channels, int offset)
{
#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                int pixelVal = img[(y * width + x) * channels + c];
                int newVal = pixelVal + offset;
                img[(y * width + x) * channels + c] = std::min(std::max(newVal, 0), 255); // Clamp between 0 and 255
            }
        }
    }
}

// Contrast Adjustment
void applyContrastAdjustment(unsigned char *img, int width, int height, int channels, float factor)
{
#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                int pixelVal = img[(y * width + x) * channels + c];
                int newVal = static_cast<int>(factor * (pixelVal - 128) + 128);
                img[(y * width + x) * channels + c] = std::min(std::max(newVal, 0), 255); // Clamp between 0 and 255
            }
        }
    }
}

// Histogram Equalization
void applyHistogramEqualization(unsigned char *img, int width, int height, int channels)
{
    int hist[256] = {0}; // Histogram of image intensities
    int cdf[256] = {0};  // Cumulative distribution function

// Calculate the histogram
#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                int pixelVal = img[(y * width + x) * channels + c];
                hist[pixelVal]++;
            }
        }
    }

    // Calculate the cumulative distribution function (CDF)
    cdf[0] = hist[0];
    for (int i = 1; i < 256; i++)
    {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    // Normalize the CDF
    int min_cdf = cdf[0];
    int max_cdf = cdf[255];
#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                int pixelVal = img[(y * width + x) * channels + c];
                int newVal = (cdf[pixelVal] - min_cdf) * 255 / (max_cdf - min_cdf);
                img[(y * width + x) * channels + c] = newVal;
            }
        }
    }
}

int main()
{
    int width, height, channels;
    unsigned char *img = stbi_load("enhance_input.jpg", &width, &height, &channels, 0);

    if (img == NULL)
    {
        cout << "Error loading image\n";
        return -1;
    }

    // Apply Brightness Correction first
    unsigned char *imgEnhanced = new unsigned char[width * height * channels];
    memcpy(imgEnhanced, img, width * height * channels);
    applyBrightnessCorrection(imgEnhanced, width, height, channels, 30); // Example offset of 30

    // Apply Contrast Adjustment next
    applyContrastAdjustment(imgEnhanced, width, height, channels, 1.5); // Example contrast factor of 1.5

    // Apply Histogram Equalization last
    applyHistogramEqualization(imgEnhanced, width, height, channels);

    // Save the final enhanced image
    stbi_write_jpg("output_enhanced.jpg", width, height, channels, imgEnhanced, 100);

    // Free the image memory
    stbi_image_free(img);
    delete[] imgEnhanced;

    return 0;
}
