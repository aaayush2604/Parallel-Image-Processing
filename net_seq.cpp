#include <iostream>
#include <cmath>
#include <cstring>
#include <string>
#include <dirent.h>
#include <direct.h>
#include <sys/stat.h>
#include <chrono>
#include <atomic>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using namespace chrono;

// Grayscale conversion
void apply_grayscale(unsigned char *img, int width, int height, int channels)
{
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

// void process_directory(const std::string &input_folder, const std::string &output_folder)
// {
//     DIR *dir = opendir(input_folder.c_str());
//     if (dir == nullptr)
//     {
//         std::cerr << "Error opening directory: " << input_folder << std::endl;
//         return;
//     }

//     struct dirent *entry;
//     while ((entry = readdir(dir)) != nullptr)
//     {
//         std::string entry_name = entry->d_name;

//         // Skip "." and ".."
//         if (entry_name == "." || entry_name == "..")
//             continue;

//         std::string input_path = input_folder + "/" + entry_name;
//         std::string output_path = output_folder + "/" + entry_name;

//         struct stat info;
//         if (stat(input_path.c_str(), &info) == 0)
//         {
//             if (S_ISDIR(info.st_mode))
//             {
//                 // If it's a directory, recursively process it
//                 _mkdir(output_path.c_str()); // Create corresponding output directory
//                 process_directory(input_path, output_path);
//             }
//             else if (S_ISREG(info.st_mode))
//             {
//                 // If it's a file, process it
//                 int width, height, channels;
//                 unsigned char *processed_img = process_image(input_path.c_str(), width, height, channels);

//                 if (processed_img != nullptr)
//                 {
//                     stbi_write_jpg(output_path.c_str(), width, height, channels, processed_img, 100);
//                     stbi_image_free(processed_img);
//                 }
//                 else
//                 {
//                     std::cerr << "Error processing image: " << input_path << std::endl;
//                 }
//             }
//         }
//     }

//     closedir(dir);
// }

// int main()
// {
//     const std::string input_folder = "melanomaDataset/melanoma_cancer_dataset"; // Replace with your input folder path
//     const std::string output_folder = "outputDataset_Seq";                      // Replace with your output folder path

//     // Create the root output directory
//     _mkdir(output_folder.c_str());

//     auto start_time = high_resolution_clock::now();
//     // Process the directory
//     process_directory(input_folder, output_folder);

//     auto end_time = high_resolution_clock::now();
//     auto duration = duration_cast<milliseconds>(end_time - start_time).count();
//     cout << "Time spent: " << duration << " ms" << endl;

//     std::cout << "Processing completed successfully!" << std::endl;
//     return 0;
// }

std::atomic<int> processed_count(0);

void process_directory(const std::string &input_folder, const std::string &output_folder, const auto &start_time)
{
    DIR *dir = opendir(input_folder.c_str());
    if (dir == nullptr)
    {
        std::cerr << "Error opening directory: " << input_folder << std::endl;
        return;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string entry_name = entry->d_name;

        // Skip "." and ".."
        if (entry_name == "." || entry_name == "..")
            continue;

        std::string input_path = input_folder + "/" + entry_name;
        std::string output_path = output_folder + "/" + entry_name;

        struct stat info;
        if (stat(input_path.c_str(), &info) == 0)
        {
            if (S_ISDIR(info.st_mode))
            {
                // If it's a directory, recursively process it
                _mkdir(output_path.c_str()); // Create corresponding output directory
                process_directory(input_path, output_path, start_time);
            }
            else if (S_ISREG(info.st_mode))
            {
                // If it's a file, process it
                int width, height, channels;
                unsigned char *processed_img = process_image(input_path.c_str(), width, height, channels);

                if (processed_img != nullptr)
                {
                    stbi_write_jpg(output_path.c_str(), width, height, channels, processed_img, 100);
                    stbi_image_free(processed_img);

                    // Increment the global counter
                    int current_count = ++processed_count;

                    // Print elapsed time for every 1000 images processed
                    if (current_count % 1000 == 0)
                    {
                        auto current_time = high_resolution_clock::now();
                        auto elapsed_time = duration_cast<milliseconds>(current_time - start_time).count();
                        std::cout << "Time spent after processing " << current_count << " images: " << elapsed_time << " ms" << std::endl;
                    }
                }
                else
                {
                    std::cerr << "Error processing image: " << input_path << std::endl;
                }
            }
        }
    }

    closedir(dir);
}

int main()
{
    const std::string input_folder = "melanomaDataset/melanoma_cancer_dataset"; // Replace with your input folder path
    const std::string output_folder = "outputDataset_Seq";                      // Replace with your output folder path

    // Create the root output directory
    _mkdir(output_folder.c_str());

    auto start_time = high_resolution_clock::now();

    // Process the directory
    process_directory(input_folder, output_folder, start_time);

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time).count();
    std::cout << "Total time spent: " << duration << " ms" << std::endl;

    std::cout << "Processing completed successfully!" << std::endl;
    return 0;
}
