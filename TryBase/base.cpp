#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

// Function to apply a subtle Gaussian blur (image smoothing)
void applyGaussianBlur(unsigned char *image, unsigned char *output, int width, int height, int channels, int kernelSize)
{
    int kernelRadius = kernelSize / 2;
    float kernel[3][3] = {{1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
                          {2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
                          {1 / 16.0f, 2 / 16.0f, 1 / 16.0f}}; // A simple 3x3 kernel for smoothing

#pragma omp parallel for
    for (int y = kernelRadius; y < height - kernelRadius; ++y)
    {
        for (int x = kernelRadius; x < width - kernelRadius; ++x)
        {
            float sum[3] = {0.0f, 0.0f, 0.0f}; // To store the sum of pixel values for R, G, B channels
            for (int ky = -kernelRadius; ky <= kernelRadius; ++ky)
            {
                for (int kx = -kernelRadius; kx <= kernelRadius; ++kx)
                {
                    int px = x + kx;
                    int py = y + ky;
                    int idx = (py * width + px) * channels;
                    for (int c = 0; c < channels; ++c)
                    {
                        sum[c] += image[idx + c] * kernel[ky + kernelRadius][kx + kernelRadius];
                    }
                }
            }
            int idx = (y * width + x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                output[idx + c] = static_cast<unsigned char>(sum[c]);
            }
        }
    }
}

// Function to apply sharpening filter (emphasizing edges)
void applySharpening(unsigned char *image, unsigned char *output, int width, int height, int channels)
{
    int kernel[3][3] = {{-1, -1, -1},
                        {-1, 9, -1},
                        {-1, -1, -1}}; // A simple sharpening kernel

#pragma omp parallel for
    for (int y = 1; y < height - 1; ++y)
    {
        for (int x = 1; x < width - 1; ++x)
        {
            float sum[3] = {0.0f, 0.0f, 0.0f}; // To store the sum of pixel values for R, G, B channels
            for (int ky = -1; ky <= 1; ++ky)
            {
                for (int kx = -1; kx <= 1; ++kx)
                {
                    int px = x + kx;
                    int py = y + ky;
                    int idx = (py * width + px) * channels;
                    for (int c = 0; c < channels; ++c)
                    {
                        sum[c] += image[idx + c] * kernel[ky + 1][kx + 1];
                    }
                }
            }
            int idx = (y * width + x) * channels;
            for (int c = 0; c < channels; ++c)
            {
                output[idx + c] = static_cast<unsigned char>(std::min(std::max(int(sum[c]), 0), 255));
            }
        }
    }
}

// Function to apply median filter for noise reduction
void applyMedianFilter(unsigned char *image, unsigned char *output, int width, int height, int channels, int kernelSize)
{
    int kernelRadius = kernelSize / 2;

#pragma omp parallel for
    for (int y = kernelRadius; y < height - kernelRadius; ++y)
    {
        for (int x = kernelRadius; x < width - kernelRadius; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                std::vector<unsigned char> neighbors;
                // Collect neighbors for the current pixel
                for (int ky = -kernelRadius; ky <= kernelRadius; ++ky)
                {
                    for (int kx = -kernelRadius; kx <= kernelRadius; ++kx)
                    {
                        int px = x + kx;
                        int py = y + ky;
                        int idx = (py * width + px) * channels;
                        neighbors.push_back(image[idx + c]);
                    }
                }
                // Sort the neighbors and select the median
                std::nth_element(neighbors.begin(), neighbors.begin() + neighbors.size() / 2, neighbors.end());
                unsigned char median = neighbors[neighbors.size() / 2];

                // Assign the median value to the output image
                int idx = (y * width + x) * channels;
                output[idx + c] = median;
            }
        }
    }
}

// Function to adjust contrast (moderate contrast enhancement)
void adjustContrast(unsigned char *image, unsigned char *output, int width, int height, int channels, float factor)
{
#pragma omp parallel for
    for (int i = 0; i < width * height * channels; ++i)
    {
        output[i] = std::min(std::max(static_cast<int>(image[i] * factor), 0), 255);
    }
}

// Function to adjust brightness (moderate brightness adjustment)
void adjustBrightness(unsigned char *image, unsigned char *output, int width, int height, int channels, int value)
{
#pragma omp parallel for
    for (int i = 0; i < width * height * channels; ++i)
    {
        output[i] = std::min(std::max(image[i] + value, 0), 255);
    }
}

int main()
{
    int width, height, channels;
    unsigned char *image = stbi_load("download.jpg", &width, &height, &channels, 0); // Load image using stb_image
    if (!image)
    {
        std::cout << "Error loading image!" << std::endl;
        return -1;
    }

    unsigned char *output = new unsigned char[width * height * channels];
    unsigned char *temp = new unsigned char[width * height * channels];

    // Apply tasks sequentially with OpenMP parallelism:
    // 1. Apply Gaussian Blur (Smoothing)
    applyGaussianBlur(image, temp, width, height, channels, 3); // Apply Gaussian blur (3x3 kernel)

    // 2. Apply Sharpening Filter (Enhance edges)
    applySharpening(temp, output, width, height, channels);

    // 3. Apply Median Filter (Noise Reduction)
    applyMedianFilter(output, temp, width, height, channels, 3); // Apply median filter (3x3 kernel)

    // 4. Adjust Contrast (Moderate)
    adjustContrast(temp, output, width, height, channels, 1.2f); // Adjust contrast

    // 5. Adjust Brightness (Moderate)
    adjustBrightness(output, temp, width, height, channels, 20); // Adjust brightness

    // Save the processed image
    stbi_write_jpg("output_image_enhanced.jpg", width, height, channels, temp, 90);

    // Cleanup
    stbi_image_free(image);
    delete[] output;
    delete[] temp;

    std::cout << "Image processing completed!" << std::endl;
    return 0;
}
