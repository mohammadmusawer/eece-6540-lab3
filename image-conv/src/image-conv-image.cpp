// Image Rotation with DPC++
//
// Author: Mohammad Musawer
//
#include <CL/sycl.hpp>
#include <cmath>
#include <array>
#include <iostream>
#include "dpc_common.hpp"
#if FPGA || FPGA_EMULATOR || FPGA_PROFILE
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;

// useful header files for image convolution
#include "utils.h"
#include "bmp-utils.h"
#include "gold.h"

using Duration = std::chrono::duration<double>;
class Timer {
 public:
  Timer() : start(std::chrono::steady_clock::now()) {}

  Duration elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start);
  }

 private:
  std::chrono::steady_clock::time_point start;
};

static const char* inputImagePath = "./Images/cat.bmp";

#define IMAGE_SIZE (720*1080)
constexpr size_t array_size = IMAGE_SIZE;
typedef std::array<float, array_size> FloatArray;

//************************************
// Image Rotation in DPC++ on device: 
//************************************
void ImageConv_v1(queue &q, float *image_in, float *image_out, float sin_theta, 
    float cos_theta, const size_t ImageRows, const size_t ImageCols) 
{

    // We create buffers for the input and output data.
    buffer<float, 1> image_in_buf(image_in, range<1>(ImageRows*ImageCols));
    buffer<float, 1> image_out_buf(image_out, range<1>(ImageRows*ImageCols));


    // Create the range object for the pixel data.
    range<2> num_items{ImageRows, ImageCols};

    // Submit a command group to the queue by a lambda function that contains the
    // data access permission and device computation (kernel).
    q.submit([&](handler &h) {
      // Create an accessor to buffers with access permission: read, write or
      // read/write. The accessor is a way to access the memory in the buffer.
      accessor srcPtr(image_in_buf, h, read_only);

      // Another way to get access is to call get_access() member function 
      auto dstPtr = image_out_buf.get_access<access::mode::write>(h);

      // Use parallel_for to run image convolution in parallel on device. This
      // executes the kernel.
      //    1st parameter is the number of work items.
      //    2nd parameter is the kernel, a lambda that specifies what to do per
      //    work item. The parameter of the lambda is the work item id.
      // DPC++ supports unnamed lambda kernel by default.
      h.parallel_for(num_items, [=](id<2> item) 
      { 

        // get row and col of the pixel assigned to this work item
        int row = item[0];
        int col = item[1];

	// Initial coordinate calculation
	int x0, y0;
	x0 = (int)(ImageRows / 2);
	y0 = ImageRows - x0;

	// x1 and y1 values 
	int x1, y1, x2, y2;
	x1 = col;
	y1 = ImageRows-row;


        // Image rotation calculation
        float rotated_x, rotated_y;

	// Rotation by an angle theta
	//rotated_x = (cos_theta * (float)(x1-x0) + sin_theta * (float)(y1-y0));
	//rotated_y = (-1 * sin_theta * (float)(x1-x0) + cos_theta * (float)(y1-y0));
	
	rotated_x = ((-1*sin_theta * (float)(y1-y0) + x0) + cos_theta * (float)(x1-x0));
	rotated_y = (sin_theta * (float)(x1-x0) + cos_theta * (float)(y1-y0) + y0);
	

	// Coordinates of a point (x1, y1) when rotated by an angle theta around (x0, y0) become (x2, y2)
	x2 = (int)rotated_x;
	y2 = ImageRows - (int)rotated_y;

        // Iterator for the filter */
        int filterIdx = 0;

        // Each work-item iterates around its local area based on the
        // size of the filter 

        float sum = 0.0f;
         
        /* Write the new pixel value and bound checking*/
	if((y2 < ImageRows) && (y2 >= 0) && (x2 < ImageCols) && (x2 >= 0)){
	
		dstPtr[y2*ImageCols+x2] = srcPtr[row*ImageCols+col];
	}
	
	//dstPtr[row*ImageCols+col] = dstPtr[y2*ImageCols+x2];

      });
  });
}


int main() {
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  ext::intel::fpga_emulator_selector d_selector;
#elif FPGA || FPGA_PROFILE
  // DPC++ extension: FPGA selector on systems with FPGA card.
  ext::intel::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  default_selector d_selector;
#endif

  int angle_theta = 45;   // Amount of degrees to rotate the image
  float pi_angle = 180.0f;
  float pi_val = 3.14159265;

  float cos_theta, sin_theta;

  cos_theta = cos(angle_theta*pi_val / pi_angle);  // Cos theta
  sin_theta = sin(angle_theta*pi_val / pi_angle);  // Sin theta

  float *hInputImage;
  float *hOutputImage;

  int imageRows;
  int imageCols;
  int i;

#ifndef FPGA_PROFILE
  // Query about the platform
  unsigned number = 0;
  auto myPlatforms = platform::get_platforms();
  // loop through the platforms to poke into
  for (auto &onePlatform : myPlatforms) {
    std::cout << ++number << " found .." << std::endl << "Platform: " 
    << onePlatform.get_info<info::platform::name>() <<std::endl;
    // loop through the devices
    auto myDevices = onePlatform.get_devices();
    for (auto &oneDevice : myDevices) {
      std::cout << "Device: " 
      << oneDevice.get_info<info::device::name>() <<std::endl;
    }
  }
  std::cout<<std::endl;
#endif

  /* Read in the BMP image */
  hInputImage = readBmpFloat(inputImagePath, &imageRows, &imageCols);
  printf("imageRows=%d, imageCols=%d\n", imageRows, imageCols);

  /* Allocate space for the output image */
  hOutputImage = (float *)malloc( imageRows*imageCols * sizeof(float) );
  for(i=0; i<imageRows*imageCols; i++)
    hOutputImage[i] = 1.0;

  Timer t;

  try {
    queue q(d_selector, dpc_common::exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";

    // Image convolution in DPC++
    ImageConv_v1(q, hInputImage, hOutputImage, sin_theta, cos_theta, imageRows, imageCols);
  } catch (exception const &e) {
    std::cout << "An exception is caught for image rotation.\n";
    std::terminate();
  }

  std::cout << t.elapsed().count() << " seconds\n";

  /* Save the output bmp */
  printf("Image was rotated %d degrees and output was saved as: cat-rotated.bmp\n", angle_theta);
  writeBmpFloat(hOutputImage, "cat-rotated.bmp", imageRows, imageCols,
          inputImagePath);

  return 0;
}
