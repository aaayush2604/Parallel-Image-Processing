#include <iostream>
using namespace std;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main()
{
    int width, height, channels;
    unsigned char *img = stbi_load("download.jpg", &width, &height, &channels, 0);

    if (img == NULL)
    {
        cout << "Error loading image\n";
        return -1;
    }

    // Process the image (e.g., apply filters, transformations)

    // Save the processed image
    stbi_write_jpg("output.jpg", width, height, channels, img, 100);

    stbi_image_free(img); // Free the image memory
    return 0;
}
