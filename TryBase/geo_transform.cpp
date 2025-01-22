#include <iostream>
#include <cmath>
#include <omp.h>
using namespace std;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Function to apply Rotation on the image
void applyRotation(unsigned char *img, unsigned char *output, int width, int height, int channels, float angle)
{
    float radians = angle * M_PI / 180.0; // Convert angle to radians
    int centerX = width / 2;
    int centerY = height / 2;

#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int newX = (int)((x - centerX) * cos(radians) - (y - centerY) * sin(radians) + centerX);
            int newY = (int)((x - centerX) * sin(radians) + (y - centerY) * cos(radians) + centerY);

            if (newX >= 0 && newX < width && newY >= 0 && newY < height)
            {
                for (int c = 0; c < channels; c++)
                {
                    output[(y * width + x) * channels + c] = img[(newY * width + newX) * channels + c];
                }
            }
        }
    }
}

// Function to apply Scaling on the image
void applyScaling(unsigned char *img, unsigned char *output, int width, int height, int channels, float scaleX, float scaleY)
{
#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int newX = (int)(x / scaleX);
            int newY = (int)(y / scaleY);

            if (newX >= 0 && newX < width && newY >= 0 && newY < height)
            {
                for (int c = 0; c < channels; c++)
                {
                    output[(y * width + x) * channels + c] = img[(newY * width + newX) * channels + c];
                }
            }
        }
    }
}

// Function to apply Horizontal Flip on the image
void applyHorizontalFlip(unsigned char *img, unsigned char *output, int width, int height, int channels)
{
#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                int oppositeX = width - 1 - x;
                // Get the original pixel and set it in the opposite location in the output
                output[(y * width + oppositeX) * channels + c] = img[(y * width + x) * channels + c];
            }
        }
    }
}

// Function to apply Vertical Flip on the image
void applyVerticalFlip(unsigned char *img, unsigned char *output, int width, int height, int channels)
{
#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                int oppositeY = height - 1 - y;
                // Get the original pixel and set it in the opposite location in the output
                output[(oppositeY * width + x) * channels + c] = img[(y * width + x) * channels + c];
            }
        }
    }
}

int main()
{
    int width, height, channels;
    unsigned char *img = stbi_load("download.jpg", &width, &height, &channels, 0);

    if (img == NULL)
    {
        cout << "Error loading image\n";
        return -1;
    }

    // Create output image buffers for processing
    unsigned char *imgRotated = new unsigned char[width * height * channels];
    unsigned char *imgScaled = new unsigned char[width * height * channels];
    unsigned char *imgHorizontallyFlipped = new unsigned char[width * height * channels];
    unsigned char *imgVerticallyFlipped = new unsigned char[width * height * channels];

    // Apply Rotation (example: 45 degrees)
    applyRotation(img, imgRotated, width, height, channels, 45);

    // Apply Scaling (example: scale 1.5x in X and 1.5x in Y direction)
    applyScaling(img, imgScaled, width, height, channels, 1.5, 1.5);

    // Apply Horizontal Flip
    applyHorizontalFlip(img, imgHorizontallyFlipped, width, height, channels);

    // Apply Vertical Flip
    applyVerticalFlip(img, imgVerticallyFlipped, width, height, channels);

    // Save the images after transformations
    stbi_write_jpg("output_rotated.jpg", width, height, channels, imgRotated, 100);
    stbi_write_jpg("output_scaled.jpg", width, height, channels, imgScaled, 100);
    stbi_write_jpg("output_horizontal_flip.jpg", width, height, channels, imgHorizontallyFlipped, 100);
    stbi_write_jpg("output_vertical_flip.jpg", width, height, channels, imgVerticallyFlipped, 100);

    // Free the image memory
    stbi_image_free(img);
    delete[] imgRotated;
    delete[] imgScaled;
    delete[] imgHorizontallyFlipped;
    delete[] imgVerticallyFlipped;

    return 0;
}
