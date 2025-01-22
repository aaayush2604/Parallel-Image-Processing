#include <iostream>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <dirent.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

// Grayscale conversion
void apply_grayscale(unsigned char *img, int width, int height, int channels)
{
#pragma omp parallel for
    for (int i = 0; i < width * height; i++)
    {
        int r = img[i * channels];
        int g = img[i * channels + 1];
        int b = img[i * channels + 2];
        unsigned char gray = 0.3 * r + 0.59 * g + 0.11 * b;
        img[i * channels] = img[i * channels + 1] = img[i * channels + 2] = gray;
    }
}

// Gaussian Blur
void apply_gaussian_blur(unsigned char *img, int width, int height, int channels)
{
    const int kernel_size = 3;
    const float kernel[3][3] = {
        {1 / 16.0, 2 / 16.0, 1 / 16.0},
        {2 / 16.0, 4 / 16.0, 2 / 16.0},
        {1 / 16.0, 2 / 16.0, 1 / 16.0}};

    unsigned char *output = new unsigned char[width * height * channels];

#pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                float sum = 0.0;
                for (int ky = -1; ky <= 1; ky++)
                {
                    for (int kx = -1; kx <= 1; kx++)
                    {
                        int pixel = img[((y + ky) * width + (x + kx)) * channels + c];
                        sum += pixel * kernel[ky + 1][kx + 1];
                    }
                }
                output[(y * width + x) * channels + c] = static_cast<unsigned char>(sum);
            }
        }
    }

    memcpy(img, output, width * height * channels);
    delete[] output;
}

void apply_sharpening(unsigned char *img, int width, int height, int channels)
{
    const int kernel_size = 3;
    const float kernel[3][3] = {
        {-1, -1, -1},
        {-1, 9, -1},
        {-1, -1, -1}};

    unsigned char *output = new unsigned char[width * height * channels];

#pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                float sum = 0.0;
                for (int ky = -1; ky <= 1; ky++)
                {
                    for (int kx = -1; kx <= 1; kx++)
                    {
                        int pixel = img[((y + ky) * width + (x + kx)) * channels + c];
                        sum += pixel * kernel[ky + 1][kx + 1];
                    }
                }
                output[(y * width + x) * channels + c] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
            }
        }
    }

    memcpy(img, output, width * height * channels);
    delete[] output;
}

// Histogram Equalization
void apply_histogram_equalization(unsigned char *img, int width, int height, int channels)
{
    int histogram[256] = {0};
    unsigned char lut[256];
    int total_pixels = width * height;

// Compute histogram
#pragma omp parallel for reduction(+ : histogram[ : 256])
    for (int i = 0; i < total_pixels; i++)
    {
        histogram[img[i * channels]]++;
    }

    // Compute cumulative distribution function (CDF)
    int cdf_min = 0;
    for (int i = 0; i < 256; i++)
    {
        if (histogram[i] > 0)
        {
            cdf_min = histogram[i];
            break;
        }
    }

    int cumulative = 0;
    for (int i = 0; i < 256; i++)
    {
        cumulative += histogram[i];
        lut[i] = static_cast<unsigned char>(round(255.0 * (cumulative - cdf_min) / (total_pixels - cdf_min)));
    }

// Apply LUT
#pragma omp parallel for
    for (int i = 0; i < total_pixels; i++)
    {
        img[i * channels] = img[i * channels + 1] = img[i * channels + 2] = lut[img[i * channels]];
    }
}

unsigned char *process_image(const char *image_path, int &width, int &height, int &channels)
{
    unsigned char *img = stbi_load(image_path, &width, &height, &channels, 0);

    if (img == NULL)
    {
        cout << "Error loading image\n";
        return nullptr;
    }

    // Apply Preprocessing Steps
    apply_grayscale(img, width, height, channels);
    apply_gaussian_blur(img, width, height, channels);
    apply_sharpening(img, width, height, channels);
    apply_histogram_equalization(img, width, height, channels);

    return img;
}

int main()
{

    // int width, height, channels;
    // unsigned char *processed_img = process_image("cancer_base copy.jpg", width, height, channels);

    // if (processed_img == nullptr)
    // {
    //     return -1;
    // }

    // stbi_write_jpg("output.jpg", width, height, channels, processed_img, 100);
    // stbi_image_free(processed_img);
    // return 0;

    const char *input_folder = "melanomaDataset/melanoma_cancer_dataset"; // Replace with your folder containing images
    const char *output_folder = "outputDataset";                          // Replace with desired output folder

    DIR *dir = opendir(input_folder);
    if (dir == nullptr)
    {
        std::cerr << "Error opening directory: " << input_folder << std::endl;
        return -1;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        if (entry->d_type == DT_REG)
        { // Check if it's a regular file
            std::string file_name(entry->d_name);
            if (file_name != "." && file_name != "..")
            {
                std::string input_image_path = std::string(input_folder) + "/" + file_name;

                int width, height, channels;
                unsigned char *processed_img = process_image(input_image_path.c_str(), width, height, channels);

                if (processed_img != nullptr)
                {
                    std::string output_image_path = std::string(output_folder) + "/" + file_name;

                    stbi_write_jpg(output_image_path.c_str(), width, height, channels, processed_img, 100);
                    stbi_image_free(processed_img);
                }
            }
        }
    }

    closedir(dir);
    return 0;
}
