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

    // Apply Contrast Adjustment and save
    unsigned char *imgContrast = new unsigned char[width * height * channels];
    memcpy(imgContrast, img, width * height * channels);
    applyContrastAdjustment(imgContrast, width, height, channels, 1.5); // Example factor of 1.5
    stbi_write_jpg("output_contrast_adjustment.jpg", width, height, channels, imgContrast, 100);

    // Apply Brightness Correction and save
    unsigned char *imgBrightness = new unsigned char[width * height * channels];
    memcpy(imgBrightness, img, width * height * channels);
    applyBrightnessCorrection(imgBrightness, width, height, channels, 30); // Example offset of 30
    stbi_write_jpg("output_brightness_correction.jpg", width, height, channels, imgBrightness, 100);

    // Apply Histogram Equalization and save
    unsigned char *imgHistogram = new unsigned char[width * height * channels];
    memcpy(imgHistogram, img, width * height * channels);
    applyHistogramEqualization(imgHistogram, width, height, channels);
    stbi_write_jpg("output_histogram_equalization.jpg", width, height, channels, imgHistogram, 100);

    // Free the image memory
    stbi_image_free(img);
    delete[] imgContrast;
    delete[] imgBrightness;
    delete[] imgHistogram;

    return 0;
}
