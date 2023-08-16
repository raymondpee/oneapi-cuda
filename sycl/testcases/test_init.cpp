#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

TEST(SYCL_TEST,test_gpu_basic) {
  // Creating buffer of 4 ints to be used inside the kernel code
  sycl::buffer<sycl::cl_int, 1> Buffer(4);

  // Creating SYCL queue
  sycl::queue Queue;

  // Size of index space for kernel
  sycl::range<1> NumOfWorkItems{Buffer.size()};

  // Submitting command group(work) to queue
  Queue.submit([&](sycl::handler &cgh) {
    // Getting write only access to the buffer on a device
    auto Accessor = Buffer.get_access<sycl::access::mode::write>(cgh);
    // Executing kernel
    cgh.parallel_for<class FillBuffer>(
        NumOfWorkItems, [=](sycl::id<1> WIid) {
          // Fill buffer with indexes
          Accessor[WIid] = (sycl::cl_int)WIid.get(0);
        });
  });

  // Getting read only access to the buffer on the host.
  // Implicit barrier waiting for queue to complete the work.
  const auto HostAccessor = Buffer.get_access<sycl::access::mode::read>();

  // Check the results
  bool MismatchFound = false;
  for (size_t I = 0; I < Buffer.size(); ++I) {
    if (HostAccessor[I] != I) {
      std::cout << "The result is incorrect for element: " << I
                << " , expected: " << I << " , got: " << HostAccessor[I]
                << std::endl;
      MismatchFound = true;
    }
  }
  EXPECT_EQ(MismatchFound, false);
}

TEST(SYCL_TEST,test_gpu_attribute) {
   // Create a SYCL device selector for GPUs
  sycl::gpu_selector selector;

  // Create a SYCL queue using the GPU device selector
  sycl::queue queue(selector);

  // Get the SYCL device associated with the queue
  sycl::device device = queue.get_device();

  auto p_name = device.get_platform().get_info<sycl::info::platform::name>();
  auto p_version = device.get_platform().get_info<sycl::info::platform::version>();
  auto d_name = device.get_info<sycl::info::device::name>();
  auto max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
  auto maxWorkItemSize1 = device.get_info<sycl::info::device::max_work_item_sizes<1>>();
  auto maxWorkItemSize2 = device.get_info<sycl::info::device::max_work_item_sizes<2>>();
  auto maxWorkItemSize3 = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
  auto max_compute_units = device.get_info<sycl::info::device::max_compute_units>();

   // Calculate the number of work groups
  auto localworkspace = (maxWorkItemSize1 + max_work_group_size - 1) / max_work_group_size;

  // Calculate the global size of the workspace
  auto globalworkspace = localworkspace * max_work_group_size;

  std::cout << "Platform Name: " << p_name << "\n";
  std::cout << "Platform Version: " << p_version << "\n";
  std::cout << "Device Name: " << d_name << "\n";
  std::cout << "Max Work Group Size: " << max_work_group_size << "\n";
  std::cout << "Max Num of Work Group: " << localworkspace << "\n";
  std::cout << "local workspace: " << localworkspace << "\n";
  std::cout << "global workspace: " << globalworkspace << "\n";

  std::cout << "Device max work item size 1: " << maxWorkItemSize1.get(0) << "\n";
  std::cout << "Device max work item size 2: " << maxWorkItemSize2.get(0) << ", " << maxWorkItemSize2.get(1) << "\n";
  std::cout << "Device max work item size 3: " << maxWorkItemSize3.get(0) << ", " << maxWorkItemSize3.get(1) << ", " << maxWorkItemSize3.get(2) << "\n";
    
  std::cout << "Max Compute Units: " << max_compute_units << "\n\n";
}