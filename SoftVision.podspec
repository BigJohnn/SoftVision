#
#  Be sure to run `pod spec lint SoftVision.podspec' to ensure this is a
#  valid spec and to remove all comments including this before submitting the spec.
#
#  To learn more about Podspec attributes see https://guides.cocoapods.org/syntax/podspec.html
#  To see working Podspecs in the CocoaPods repo see https://github.com/CocoaPods/Specs/
#

Pod::Spec.new do |spec|

  # ―――  Spec Metadata  ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #
  #
  #  These will help people to find your library, and whilst it
  #  can feel like a chore to fill in it's definitely to your advantage. The
  #  summary should be tweet-length, and the description more in depth.
  #

  spec.name         = "SoftVision"
  spec.version      = "0.0.1"
  spec.summary      = "A soft vision framework for mobile devices."

  # This description is used to generate tags and improve search results.
  #   * Think: What does it do? Why did you write it? What is the focus?
  #   * Try to keep it short, snappy and to the point.
  #   * Write the description between the DESC delimiters below.
  #   * Finally, don't worry about the indent, CocoaPods strips it!
  spec.description  = "Fast with essential functions, to reconstruct your space around."

  spec.homepage     = "https://github.com/BigJohnn/SoftVision.git"
  # spec.screenshots  = "www.example.com/screenshots_1.gif", "www.example.com/screenshots_2.gif"


  # ―――  Spec License  ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #
  #
  #  Licensing your code is important. See https://choosealicense.com for more info.
  #  CocoaPods will detect a license file if there is a named LICENSE*
  #  Popular ones are 'MIT', 'BSD' and 'Apache License, Version 2.0'.
  #

  spec.license      = { :type => "GPLv3", :file => "LICENSE" }


  # ――― Author Metadata  ――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #
  #
  #  Specify the authors of the library, with email addresses. Email addresses
  #  of the authors are extracted from the SCM log. E.g. $ git log. CocoaPods also
  #  accepts just a name if you'd rather not provide an email address.
  #
  #  Specify a social_media_url where others can refer to, for example a twitter
  #  profile URL.
  #

  spec.author             = { "BigJohhn" => "kjustdoitno1@gmail.com" }

  # ――― Platform Specifics ――――――――――――――――――――――――――――――――――――――――――――――――――――――― #
  #
  #  If this Pod runs only on iOS or OS X, then specify the platform and
  #  the deployment target. You can optionally include the target after the platform.
  #

  spec.ios.deployment_target = "12.0"
  spec.source       = { :git => "https://github.com/BigJohnn/SoftVision.git", :tag => "#{spec.version}"}



  # ――― Source Code ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #
  #
  #  CocoaPods is smart about how it includes source code. For source files
  #  giving a folder will include any swift, h, m, mm, c & cpp files.
  #  For header files it will include any header in the folder.
  #  Not including the public_header_files will make all headers public.
  #

  spec.source_files  = "src/*.{h,hpp,cpp,metal}"
                        
#  spec.source_files  = "Classes", "Classes/**/*.{h,cpp}"
#  spec.exclude_files = "src/*test.{h,hpp,cpp}"

#  spec.public_header_files = "*.{h}"

  spec.subspec 'feature' do |ss|
      ss.source_files = "src/feature/**/*.{h,hpp,cpp}"
#      ss.exclude_files = "src/feature/*test.{h,hpp,cpp}"
      ss.dependency 'VLFeat'
#      ss.frameworks = 'frameworks'
  end
#
  spec.subspec 'featureEngine' do |ss|
      ss.source_files = "src/featureEngine/*.{h,hpp,cpp}"

#      ss.dependency 'SoftVision/feature'
  end
##
  spec.subspec 'numeric' do |ss|
      ss.source_files = "src/numeric/*.{h,hpp,cpp}"
#      ss.exclude_files = "src/numeric/*test.{h,hpp,cpp}"
#      ss.libraries = 'stdc++'
#      ss.dependency 'Eigen'
  end
##
  spec.subspec 'image' do |ss|
      ss.source_files = "src/image/*.{hpp,cpp}"
#      "src/image/convolution*.{hpp,cpp}",
#      "src/image/filtering.{hpp,cpp}",
#      "src/image/pixelTypes.hpp"
  end
##
  spec.subspec 'sfmData' do |ss|
      ss.source_files = "src/sfmData/**/*.{hpp,cpp}"
#      ss.dependency 'Eigen'
#      ss.exclude_files = "src/sfmData/*test.{h,hpp,cpp}"
  end
  
  spec.subspec 'sfmDataIO' do |ss|
      ss.source_files = "src/sfmDataIO/**/*.{hpp,cpp}"
  end
  
  spec.subspec 'sfm' do |ss|
      ss.source_files = "src/sfm/**/*.{hpp,cpp}"
  end
  
  spec.subspec 'common' do |ss|
      ss.source_files = "src/common/*.{h}"
  end
  
  spec.subspec 'camera' do |ss|
      ss.source_files = "src/camera/*.{hpp,cpp}"
  end
#
  spec.subspec 'geometry' do |ss|
      ss.source_files = "src/geometry/*.{hpp,cpp}"
#      ss.exclude_files = "src/geometry/*test.{h,hpp,cpp}"
  end
#  
  spec.subspec 'system' do |ss|
      ss.source_files = "src/system/*.{h,hpp,cpp}"
#      ss.exclude_files = "src/system/Progress*.*"
  end

#  spec.ios.resource_bundle = { 'nonFree' => 'src/nonFree/**/*.{hpp,cpp,h,c}' }
  
  spec.subspec 'matching' do |ss|
      ss.source_files = "src/matching/*.{hpp,cpp}"
  end
  
  spec.subspec 'stl' do |ss|
      ss.source_files = "src/stl/*.{h,hpp,cpp}"
  end
  
  spec.subspec 'utils' do |ss|
      ss.source_files = "src/utils/*.*"
  end

  spec.subspec 'vectorGraphics' do |ss|
      ss.source_files = "src/vectorGraphics/*.hpp"
  end
  
  spec.subspec 'imageMatching' do |ss|
      ss.source_files = "src/imageMatching/*.{hpp,cpp}"
  end
  
  spec.subspec 'voctree' do |ss|
      ss.source_files = "src/voctree/*.{hpp,cpp}"
      ss.exclude_files = "src/voctree/*Loader.*"
  end

  spec.subspec 'matchingImageCollection' do |ss|
      ss.source_files = "src/matchingImageCollection/*.{hpp,cpp}"
  end
  
  spec.subspec 'multiview' do |ss|
      ss.source_files = "src/multiview/**/*.{hpp,cpp}"
  end
  
  spec.subspec 'track' do |ss|
      ss.source_files = "src/track/**/*.{hpp,cpp}"
  end
  
  spec.subspec 'robustEstimation' do |ss|
      ss.source_files = "src/robustEstimation/**/*.{hpp,cpp}"
  end
  
  spec.subspec 'boost' do |ss|
      ss.source_files = "src/boost/**/*.{hpp,cpp}"
  end
  
  spec.subspec 'graph' do |ss|
      ss.source_files = "src/graph/**/*.{hpp,cpp}"
  end
  
  spec.subspec 'htmlDoc' do |ss|
      ss.source_files = "src/htmlDoc/*.{hpp}"
  end
  
  spec.subspec 'linearProgramming' do |ss|
      ss.source_files = "src/linearProgramming/*.{hpp, cpp}"
  end
  
  spec.subspec 'mvsData' do |ss|
      ss.source_files = "src/mvsData/*.{hpp,cpp}"
  end
  
  spec.subspec 'mvsUtils' do |ss|
      ss.source_files = "src/mvsUtils/*.{hpp,cpp}"
  end
  
  spec.subspec 'depthMap' do |ss|
      ss.source_files = "src/depthMap/**/*.{hpp,mm,cpp,metal}"
  end
  
  spec.subspec 'gpu' do |ss|
      ss.source_files = "src/gpu/*.{hpp,cpp,mm}"
  end
  
  # ――― Resources ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #
  #
  #  A list of resources included with the Pod. These are copied into the
  #  target bundle with a build phase script. Anything else will be cleaned.
  #  You can preserve files from being cleaned, please don't preserve
  #  non-essential files like tests, examples and documentation.
  #

  # spec.resource  = "icon.png"
  # spec.resources = "Resources/*.png"

  # spec.preserve_paths = "FilesToSave", "MoreFilesToSave"


  # ――― Project Linking ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #
  #
  #  Link your library with frameworks, or libraries. Libraries do not include
  #  the lib prefix of their name.
  #

  # spec.framework  = "SomeFramework"
  # spec.frameworks = "SomeFramework", "AnotherFramework"

  # spec.library   = "iconv"
  # spec.libraries = "iconv", "xml2"


  # ――― Project Settings ――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #
  #
  #  If your library depends on compiler flags you can set them in the xcconfig hash
  #  where they will only apply to your library. If you depend on other Podspecs
  #  you can include multiple dependencies to ensure it works.

  # spec.requires_arc = true

  spec.header_mappings_dir = 'src'
#   spec.xcconfig = { "HEADER_SEARCH_PATHS" => "$(SDKROOT)/usr/include/libxml2" }
  spec.xcconfig = { #'CLANG_CXX_LIBRARY' => 'libstdc++',
  'HEADER_SEARCH_PATHS' => '${PROJECT_DIR}/../deps/eigen3 ${PROJECT_DIR}/../deps/OpenImageIO/include $(PROJECT_DIR)/Headers/Public/SoftVision', # To make angled quotes recursive.
  }
  
#  spec.compiler_flags = '-DEIGEN_MAX_STATIC_ALIGN_BYTES=0 -DEIGEN_MAX_ALIGN_BYTES=0'
  spec.compiler_flags = '-DEIGEN_MAX_STATIC_ALIGN_BYTES=0 -DEIGEN_MAX_ALIGN_BYTES=0 -DVL_DISABLE_SSE2'
#   spec.dependency "JSONKit", "~> 1.4"

  spec.dependency "Alembic"
  spec.dependency "assimp"
  spec.dependency "glog"
#  spec.dependency "coin"
  spec.dependency "Eigen"
  spec.dependency "libpng"
  spec.dependency "libyuv"
  spec.dependency "openmp"
  spec.dependency "lemon"
  spec.dependency "Ceres"
  spec.dependency "cJSON"
  
  
  spec.public_header_files = 'src/depthMap/gpu/**/*.metal'
  
  spec.libraries             = 'stdc++'
  
  spec.ios.resource_bundle = { 'vocabulary' => 'voc', 'metalshaders' => 'src/depthMap/gpu/device'  }

end
