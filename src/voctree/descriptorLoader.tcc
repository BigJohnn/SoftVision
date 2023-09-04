// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <feature/Descriptor.hpp>
#include <system/ProgressDisplay.hpp>
#include <SoftVisionLog.h>
#include <cstdio>

//#include <boost/filesystem.hpp>
//#include <boost/algorithm/string/case_conv.hpp>

#include <iostream>
#include <fstream>

namespace voctree {

template<class DescriptorT, class FileDescriptorT>
std::size_t readDescFromFiles(const sfmData::SfMData& sfmData,
                         const std::vector<std::string>& featuresFolders,
                         std::vector<DescriptorT>& descriptors,
                         std::vector<std::size_t> &numFeatures)
{
//  namespace bfs = boost::filesystem;
  std::map<IndexT, std::string> descriptorsFiles;
  getListOfDescriptorFiles(sfmData, featuresFolders, descriptorsFiles);
  std::size_t numDescriptors = 0;

  // Allocate the memory by reading in a first time the files to get the number
  // of descriptors
  int bytesPerElement = 0;

  // Display infos and progress bar
  LOG_DEBUG("Pre-computing the memory needed...");
  auto display = system2::createConsoleProgressDisplay(descriptorsFiles.size(), std::cout);

  // Read all files and get the number of descriptors to load
  for(const auto &currentFile : descriptorsFiles)
  {
    // if it is the first one read the number of descriptors and the type of data (we are assuming the the feat are all the same...)
    // bytesPerElement could be 0 even after the first element (eg it has 0 descriptors...), so do it until we get the correct info
    if(bytesPerElement == 0)
    {
      getInfoBinFile(currentFile.second, DescriptorT::static_size, numDescriptors, bytesPerElement);
    }
    else
    {
        //TODO: test
        FILE* fp = fopen(currentFile.second.c_str(), "rb");
        fseek(fp, 0, SEEK_END); // seek to end of file
        int file_size = ftell(fp); // get current file pointer
        fseek(fp, 0, SEEK_SET);
        fclose(fp);
      // get the file size in byte and estimate the number of features without opening the file
      numDescriptors += (file_size / bytesPerElement) / DescriptorT::static_size;
    }
    ++display;
  }
  LOG_DEBUG("Found %lu descriptors overall, allocating memory...", numDescriptors);
  if(bytesPerElement == 0)
  {
    LOG_INFO("No descriptor file found");
    return 0;
  }

  // Allocate the memory
  descriptors.reserve(numDescriptors);
  std::size_t numDescriptorsCheck = numDescriptors; // for later check
  numDescriptors = 0;

  // Read the descriptors
  LOG_DEBUG("Reading the descriptors...");
  display.restart(descriptorsFiles.size());

  // Run through the path vector and read the descriptors
  for(const auto &currentFile : descriptorsFiles)
  {
    // Read the descriptors and append them in the vector
    feature::loadDescsFromBinFile<DescriptorT, FileDescriptorT>(currentFile.second, descriptors, true);
    std::size_t result = descriptors.size();

    // Add the number of descriptors from this file
    numFeatures.push_back(result - numDescriptors);

    // Update the overall counter
    numDescriptors = result;

    ++display;
  }
  assert(numDescriptors == numDescriptorsCheck);

  // Return the result
  return numDescriptors;
}

} // namespace voctree
