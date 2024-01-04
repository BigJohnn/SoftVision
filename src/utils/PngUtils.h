//
//  PngUtils.h
//  Pods
//
//  Created by HouPeihong on 2023/8/26.
//

#ifndef PngUtils_h
#define PngUtils_h

#include "png.h"
#include <SoftVisionLog.h>

void loadpng(std::vector<uint8_t>& image_buffer, const char* file_path, int& image_width, int& image_height)
{
    FILE *fp = fopen(file_path, "rb");
    if (!fp)
    {
        LOG_ERROR("PNG file cannot open for read!! %s",file_path);
        return;
    }

      png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(!png) {
        LOG_ERROR("png_create_read_struct FAILED.");
    }

      png_infop info = png_create_info_struct(png);
    if(!info) {
        LOG_ERROR("png_create_info_struct FAILED.");
    }

      if(setjmp(png_jmpbuf(png))) abort();

      png_init_io(png, fp);

      png_read_info(png, info);

      image_width      = png_get_image_width(png, info);
      image_height     = png_get_image_height(png, info);
      png_byte color_type = png_get_color_type(png, info);
      int bit_depth  = png_get_bit_depth(png, info);

      // Read any color_type into 8bit depth, RGBA format.
      // See http://www.libpng.org/pub/png/libpng-manual.txt

      if(bit_depth == 16)
        png_set_strip_16(png);

      if(color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

      // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
      if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

      if(png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

      // These color_type don't have an alpha channel then fill it with 0xff.
      if(color_type == PNG_COLOR_TYPE_RGB ||
         color_type == PNG_COLOR_TYPE_GRAY ||
         color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

      if(color_type == PNG_COLOR_TYPE_GRAY ||
         color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

      png_read_update_info(png, info);

//      if (row_pointers) abort();
    png_bytep* row_pointers = nullptr;

      row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * image_height);
    image_buffer.resize(image_height * image_width * 4);
      for(int y = 0; y < image_height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
//          image_buffer = row_pointers[y] - (image_height - 1 - y) * image_width * 4;
          
      }
    
//    for(int k = 0; k < image_height; k++)
//    {
//        row_pointers[k] =
//            (const_cast<uint8_t *>(image_buffer)) +
//            (image_height - 1 - k) *
//            image_width * 4;
//    }

      png_read_image(png, row_pointers);

    for(int y = 0; y < image_height; y++) {
        memcpy(image_buffer.data() + (image_height - 1 - y) * image_width * 4, row_pointers[y], image_width * 4);
    }
      fclose(fp);

      png_destroy_read_struct(&png, &info, NULL);
    
    free(row_pointers);
    
}

void write2png(const char* output_path, int image_width, int image_height, const uint8_t* image_buffer)
{
    FILE *fp = fopen(output_path, "wb");
    if (!fp)
    {
        LOG_ERROR("PNG file cannot open for write!!");
    }
    
    png_structp png_ptr = png_create_write_struct
    (PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    
    if (!png_ptr)
        LOG_ERROR("PNG ptr null!");
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_write_struct(&png_ptr,
        (png_infopp)NULL);
            LOG_ERROR("info ptr null!");
    }
    
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        LOG_ERROR("Error during init_io!");
    }
    
    png_init_io(png_ptr, fp);
    
//        void write_row_callback(png_ptr, png_uint_32 row,
//        int pass);
//        {
//        /* put your code here */
//        }
//        png_set_write_status_fn(png_ptr, write_row_callback);
    
    /* turn on or off filtering, and/or choose
    specific filters. You can use either a single
    PNG_FILTER_VALUE_NAME or the bitwise OR of one
    or more PNG_FILTER_NAME masks. */
//        png_set_filter(png_ptr, 0,
//        PNG_FILTER_NONE | PNG_FILTER_VALUE_NONE |
//        PNG_FILTER_SUB | PNG_FILTER_VALUE_SUB |
//        PNG_FILTER_UP | PNG_FILTER_VALUE_UP |
//        PNG_FILTER_AVG | PNG_FILTER_VALUE_AVG |
//        PNG_FILTER_PAETH | PNG_FILTER_VALUE_PAETH|
//        PNG_ALL_FILTERS);
    
    png_set_IHDR(png_ptr, info_ptr, image_width, image_height,
    8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    
    png_write_info(png_ptr, info_ptr);
    
    png_set_bgr(png_ptr);
    
    png_bytep *row_pointers = new png_bytep [image_height];
        
    for(int k = 0; k < image_height; k++)
    {
        row_pointers[k] =
            (const_cast<uint8_t *>(image_buffer)) +
            (image_height - 1 - k) *
            image_width * 4;
    }
    
    /* write out the entire image data in one call */
    png_write_image(png_ptr, row_pointers);

    /* It is REQUIRED to call this to finish writing the rest of the file */
    png_write_end(png_ptr, info_ptr);
    
    /* clean up after the write, and free any memory allocated */
    png_destroy_write_struct(&png_ptr, &info_ptr);

    delete [] row_pointers;
    
    fclose(fp);
//        png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_BGR, NULL);
}
#endif /* PngUtils_h */
