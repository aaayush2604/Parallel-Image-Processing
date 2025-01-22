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

// Apply Sobel Edge Detection filter
void applySobelEdgeDetection(unsigned char *img, int width, int height, int channels)
{
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    unsigned char *temp = new unsigned char[width * height * channels];

#pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                int gradX = 0, gradY = 0;
                for (int ky = -1; ky <= 1; ky++)
                {
                    for (int kx = -1; kx <= 1; kx++)
                    {
                        int pixelVal = img[((y + ky) * width + (x + kx)) * channels + c];
                        gradX += pixelVal * Gx[ky + 1][kx + 1];
                        gradY += pixelVal * Gy[ky + 1][kx + 1];
                    }
                }
                int magnitude = static_cast<int>(sqrt(gradX * gradX + gradY * gradY));
                temp[(y * width + x) * channels + c] = std::min(magnitude, 255); // Clip to 255
            }
        }
    }

    // Copy the processed image back
    memcpy(img, temp, width * height * channels);

    delete[] temp;
}

void applyGaussianBlur(unsigned char *img, int width, int height, int channels)
{
    // Stronger, normalized 21x21 Gaussian kernel
    const int KERNEL_SIZE = 21; // Kernel size
    const int OFFSET = KERNEL_SIZE / 2;
    float kernel[KERNEL_SIZE][KERNEL_SIZE];

    // Generate a Gaussian kernel programmatically (standard deviation = 5.0)
    float sigma = 5.0f;
    float sum = 0.0f;
    for (int y = 0; y < KERNEL_SIZE; y++)
    {
        for (int x = 0; x < KERNEL_SIZE; x++)
        {
            float expVal = -((x - OFFSET) * (x - OFFSET) + (y - OFFSET) * (y - OFFSET)) / (2 * sigma * sigma);
            kernel[y][x] = exp(expVal);
            sum += kernel[y][x];
        }
    }

    // Normalize the kernel
    for (int y = 0; y < KERNEL_SIZE; y++)
    {
        for (int x = 0; x < KERNEL_SIZE; x++)
        {
            kernel[y][x] /= sum;
        }
    }

    unsigned char *temp = new unsigned char[width * height * channels];

#pragma omp parallel for collapse(2)
    for (int y = OFFSET; y < height - OFFSET; y++) // Account for kernel size
    {
        for (int x = OFFSET; x < width - OFFSET; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                float newVal = 0.0f;
                for (int ky = -OFFSET; ky <= OFFSET; ky++)
                {
                    for (int kx = -OFFSET; kx <= OFFSET; kx++)
                    {
                        int pixelVal = img[((y + ky) * width + (x + kx)) * channels + c];
                        newVal += pixelVal * kernel[ky + OFFSET][kx + OFFSET];
                    }
                }
                temp[(y * width + x) * channels + c] = std::min(std::max(static_cast<int>(newVal), 0), 255);
            }
        }
    }

    memcpy(img, temp, width * height * channels);
    delete[] temp;
}

// Apply Sharpening filter
void applySharpeningFilter(unsigned char *img, int width, int height, int channels)
{
    // Reduced sharpening kernel
    double kernel[3][3] = {{0, -0.5, 0}, {-0.5, 5, -0.5}, {0, -0.5, 0}}; // Sharpen kernel with a reduced effect

    unsigned char *temp = new unsigned char[width * height * channels];

#pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                int newVal = 0;
                for (int ky = -1; ky <= 1; ky++)
                {
                    for (int kx = -1; kx <= 1; kx++)
                    {
                        int pixelVal = img[((y + ky) * width + (x + kx)) * channels + c];
                        newVal += pixelVal * kernel[ky + 1][kx + 1];
                    }
                }
                temp[(y * width + x) * channels + c] = std::min(std::max(newVal, 0), 255);
            }
        }
    }

    memcpy(img, temp, width * height * channels);
    delete[] temp;
}

int main()
{
    int width, height, channels;
    unsigned char *img = stbi_load("Filter_input.jpg", &width, &height, &channels, 0);

    if (img == NULL)
    {
        cout << "Error loading image\n";
        return -1;
    }

    // Apply Gaussian Blur and save
    unsigned char *imgGaussian = new unsigned char[width * height * channels];
    memcpy(imgGaussian, img, width * height * channels);
    applyGaussianBlur(imgGaussian, width, height, channels);
    stbi_write_jpg("output_gaussian_blur.jpg", width, height, channels, imgGaussian, 100);

    // Apply Sobel Edge Detection and save
    unsigned char *imgSobel = new unsigned char[width * height * channels];
    memcpy(imgSobel, img, width * height * channels);
    applySobelEdgeDetection(imgSobel, width, height, channels);
    stbi_write_jpg("output_sobel_edge_detection.jpg", width, height, channels, imgSobel, 100);

    // Apply Sharpening and save
    unsigned char *imgSharpen = new unsigned char[width * height * channels];
    memcpy(imgSharpen, img, width * height * channels);
    applySharpeningFilter(imgSharpen, width, height, channels);
    stbi_write_jpg("output_sharpening.jpg", width, height, channels, imgSharpen, 100);

    // Free the image memory
    stbi_image_free(img);
    delete[] imgGaussian;
    delete[] imgSobel;
    delete[] imgSharpen;

    return 0;
}
