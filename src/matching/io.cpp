// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "io.hpp"
#include <matching/IndMatch.hpp>
//#include <config.hpp>
#include <SoftVisionLog.h>

//#include <boost/filesystem.hpp>
//#include <boost/range/iterator_range.hpp>

#include <map>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <unistd.h>
#include <utils/strUtils.hpp>
#include <dirent.h>
//namespace fs = boost::filesystem;


namespace matching {

bool LoadMatchFile(PairwiseMatches& matches, const std::string& filepath)
{
  const std::string ext = utils::GetFileExtension(filepath);
//  if(!fs::exists(filepath))
  if(0 != access(filepath.c_str(), F_OK))
    return false;

  if(ext == ".txt")
  {
    std::ifstream stream(filepath);
    if (!stream.is_open())
      return false;

    // Read from the text file
    // I J
    // nbDescType
    // descType matchesCount
    // idx idx
    // ...
    // descType matchesCount
    // idx idx
    // ...
    std::size_t I = 0;
    std::size_t J = 0;
    std::size_t nbDescType = 0;
    while(stream >> I >> J >> nbDescType)
    {
      for(std::size_t i = 0; i < nbDescType; ++i)
      {
        std::string descTypeStr;
        std::size_t nbMatches = 0;
        // Read descType and number of matches
        stream >> descTypeStr >> nbMatches;

        feature::EImageDescriberType descType = feature::EImageDescriberType_stringToEnum(descTypeStr);
        std::vector<IndMatch> matchesPerDesc(nbMatches);
        // Read all matches
        for(std::size_t i = 0; i < nbMatches; ++i)
        {
          stream >> matchesPerDesc[i];
        }
        matches[std::make_pair(I,J)][descType] = std::move(matchesPerDesc);
      }
    }
    stream.close();
    return true;
  }
  else
  {
    LOG_INFO("Unknown matching file format: %s", ext.c_str());
  }
  return false;
}

void filterMatchesByViews(PairwiseMatches& matches, const std::set<IndexT>& viewsKeys)
{
  matching::PairwiseMatches filteredMatches;
  for (matching::PairwiseMatches::const_iterator iter = matches.begin();
    iter != matches.end();
    ++iter)
  {
    if(viewsKeys.find(iter->first.first) != viewsKeys.end() &&
       viewsKeys.find(iter->first.second) != viewsKeys.end())
    {
      filteredMatches.insert(*iter);
    }
  }
  matches.swap(filteredMatches);
}

void filterTopMatches(PairwiseMatches& allMatches,  int maxNum, int minNum)
{
  if (maxNum <= 0 && minNum <=0)
    return;
  if (maxNum > 0 && minNum > maxNum)
    throw std::runtime_error("The minimum number of matches is higher than the maximum.");

  for(auto& matchesPerDesc: allMatches)
  {
    for(auto& matches: matchesPerDesc.second)
    {
      IndMatches& m = matches.second;
      if (minNum > 0 && m.size() < minNum)
        m.clear();
      else if (maxNum > 0 && m.size() > maxNum)
        m.erase(m.begin()+ maxNum, m.end());
    }
  }
}

void filterMatchesByDesc(PairwiseMatches& allMatches, const std::vector<feature::EImageDescriberType>& descTypesFilter)
{
  matching::PairwiseMatches filteredMatches;
  for(const auto& matchesPerDesc: allMatches)
  {
    for(const auto& matches: matchesPerDesc.second)
    {
      const IndMatches& m = matches.second;
      // if current descType in descTypesFilter
      if(std::find(descTypesFilter.begin(), descTypesFilter.end(), matches.first) != descTypesFilter.end())
        filteredMatches[matchesPerDesc.first][matches.first] = m;
    }
  }
  allMatches.swap(filteredMatches);
}

std::size_t LoadMatchFilePerImage(PairwiseMatches& matches,
                                  const std::set<IndexT>& viewsKeys,
                                  const std::string& folder,
                                  const std::string& extension)
{
  int nbLoadedMatchFiles = 0;
  // Load one match file per image
  #pragma omp parallel for num_threads(3)
  for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(viewsKeys.size()); ++i)
  {
    std::set<IndexT>::const_iterator it = viewsKeys.begin();
    std::advance(it, i);
    const IndexT idView = *it;
    const std::string matchFilename = std::to_string(idView) + "." + extension;
    PairwiseMatches fileMatches;
    if(!LoadMatchFile(fileMatches, folder + matchFilename ))
    {
      #pragma omp critical
      {
        LOG_DEBUG("Unable to load match file: %s in: %s",matchFilename.c_str(),folder.c_str());
      }
      continue;
    }
    #pragma omp critical
    {
      ++nbLoadedMatchFiles;
      // merge the loaded matches into the output
      for(const auto& v: fileMatches)
      {
        matches[v.first] = v.second;
      }
    }
  }
  return nbLoadedMatchFiles;
}

/**
 * Load and add pair-wise matches to \p matches from all files in \p folder matching \p pattern.
 * @param[out] matches PairwiseMatches to add loaded matches to
 * @param[in] folder Folder to load matches files from
 * @param[in] pattern Pattern that files must respect to be loaded
 */
std::size_t loadMatchesFromFolder(PairwiseMatches& matches, const std::string& folder, const std::string& pattern)
{
  std::size_t nbLoadedMatchFiles = 0;
  std::vector<std::string> matchFiles;
  // list all matches files in 'folder' matching (i.e containing) 'pattern',
    //TODO: check!
    struct dirent *entry;
    DIR *dp;

    dp = opendir(folder.c_str());
    if (dp == NULL) {
        perror("opendir: Path does not exist or could not be read.");
        return 0;
    }

    std::vector<std::string> files;
    while ((entry = readdir(dp)))
        files.emplace_back(entry->d_name);

    closedir(dp);
    
  for(const auto& entry : files)
  {
    if(entry.find(pattern) != std::string::npos)
    {
      matchFiles.push_back(entry);
    }
  }
    

  #pragma omp parallel for num_threads(3)
  for(int i = 0; i < matchFiles.size(); ++i)
  {
    const std::string& matchFile = matchFiles[i];
    PairwiseMatches fileMatches;
    LOG_DEBUG("Loading match file: %s" , matchFile.c_str());
    if(!LoadMatchFile(fileMatches, matchFile))
    {

      LOG_X("Unable to load match file: " << matchFile);
      continue;
    }
    #pragma omp critical
    {
    for(const auto& matchesPerView: fileMatches)
    {
      const Pair& pair = matchesPerView.first;
      const MatchesPerDescType& pairMatches = matchesPerView.second;
      for(const auto& matchesPerDescType : pairMatches)
      {
        const feature::EImageDescriberType& descType = matchesPerDescType.first;
        const auto& pairMatches = matchesPerDescType.second;
        // merge in global map
        std::copy(
          std::make_move_iterator(pairMatches.begin()), 
          std::make_move_iterator(pairMatches.end()), 
          std::back_inserter(matches[pair][descType])
        );
      }
    }
    ++nbLoadedMatchFiles;
    }   
  }
  if(!nbLoadedMatchFiles)
      LOG_X("No matches file loaded in: " << folder);
  return nbLoadedMatchFiles;
}

bool Load(PairwiseMatches& matches,
          const std::set<IndexT>& viewsKeysFilter,
          const std::vector<std::string>& folders,
          const std::vector<feature::EImageDescriberType>& descTypesFilter,
          int maxNbMatches,
          int minNbMatches)
{
  std::size_t nbLoadedMatchFiles = 0;
  const std::string pattern = "matches.txt";

  // build up a set with normalized paths to remove duplicates
  std::set<std::string> foldersSet;
  for(const auto& folder : folders)
  {
    if(0 == access(folder.c_str(), F_OK))
    {
//      foldersSet.insert(fs::canonical(folder).string());
        foldersSet.insert(folder);
    }
  }

  for(const auto& folder : foldersSet)
  {
    nbLoadedMatchFiles += loadMatchesFromFolder(matches, folder, pattern);
  }

  if(!nbLoadedMatchFiles)
    return false;

  const auto logMatches = [](const PairwiseMatches& m)
  {
    for(const auto& imagePairIt: m)
    {
      std::stringstream ss;
      ss << " * " << imagePairIt.first.first << "-" << imagePairIt.first.second << ": " << imagePairIt.second.getNbAllMatches() << "    ";
      for(const auto& matchesPerDeskIt: imagePairIt.second)
         ss << " [" << feature::EImageDescriberType_enumToString(matchesPerDeskIt.first) << ": " << matchesPerDeskIt.second.size() << "]";
        LOG_X(ss.str());
    }
  };

  LOG_INFO("Matches per image pair (before filtering):");
  logMatches(matches);

  if(!viewsKeysFilter.empty())
    filterMatchesByViews(matches, viewsKeysFilter);

  if(!descTypesFilter.empty())
    filterMatchesByDesc(matches, descTypesFilter);

  filterTopMatches(matches, maxNbMatches, minNbMatches);

  LOG_INFO("Matches per image pair (after filtering):");
  logMatches(matches);

  return true;
}

class MatchExporter
{
private:
  void saveTxt(
    const std::string& filepath,
    const PairwiseMatches::const_iterator& matchBegin,
    const PairwiseMatches::const_iterator& matchEnd)
  {
//    const fs::path bPath = fs::path(filepath);
//    const std::string tmpPath = (bPath.parent_path() / bPath.stem()).string() + "." + fs::unique_path().string() + bPath.extension().string();

    // write temporary file
    {
      std::ofstream stream(filepath, std::ios::out);
      for(PairwiseMatches::const_iterator match = matchBegin;
        match != matchEnd;
        ++match)
      {
        const std::size_t I = match->first.first;
        const std::size_t J = match->first.second;
        const MatchesPerDescType & matchesPerDesc = match->second;
        stream << I << " " << J << '\n'
               << matchesPerDesc.size() << '\n';
        for(const auto& m: matchesPerDesc)
        {
          stream << feature::EImageDescriberType_enumToString(m.first) << " " << m.second.size() << '\n';
          copy(m.second.begin(), m.second.end(), std::ostream_iterator<IndMatch>(stream, "\n"));
        }
      }
    }

    // rename temporary file
//    fs::rename(tmpPath, filepath);
  }

public:
  MatchExporter(
    const PairwiseMatches& matches,
    const std::string& folder,
    const std::string& filename)
    : m_matches(matches)
    , m_directory(folder)
    , m_filename(filename)
  {}

  ~MatchExporter() = default;
  
  void saveGlobalFile()
  {
    const std::string filepath = m_directory + m_filename;

    if(m_ext == ".txt")
      saveTxt(filepath, m_matches.begin(), m_matches.end());
    else
      throw std::runtime_error(std::string("Unknown matching file format: ") + m_ext);
  }

  /// Export matches into separate files, one for each image.
  void saveOneFilePerImage()
  {
    std::set<IndexT> keys;
    std::transform(
        m_matches.begin(), m_matches.end(),
        std::inserter(keys, keys.begin()),
        [](const PairwiseMatches::value_type &v) { return v.first.first; });

    PairwiseMatches::const_iterator matchBegin = m_matches.begin();
    PairwiseMatches::const_iterator matchEnd = m_matches.end();
    for(IndexT key: keys)
    {
      PairwiseMatches::const_iterator match = matchBegin;
      while(match != matchEnd && match->first.first == key)
        ++match;
//      const std::string filepath = (fs::path(m_directory) / (std::to_string(key) + "." + m_filename)).string();
        const std::string filepath = m_directory + "/" + std::to_string(key) + m_filename;
      LOG_DEBUG("Export Matches in: %s", filepath.c_str());
      
      if(m_ext == ".txt")
        saveTxt(filepath, matchBegin, match);
      else
        throw std::runtime_error(std::string("Unknown matching file format: ") + m_ext);

      matchBegin = match;
    }
  }

public:
  const PairwiseMatches& m_matches;
  const std::string m_ext = ".txt";
  std::string m_directory;
  std::string m_filename;
};

bool Save(const PairwiseMatches& matches,
          const std::string& folder,
          const std::string& extension,
          bool matchFilePerImage,
          const std::string& prefix)
{
  const std::string filename = prefix + "matches." + extension;
  MatchExporter exporter(matches, folder, filename);

  if(matchFilePerImage)
    exporter.saveOneFilePerImage();
  else
    exporter.saveGlobalFile();

  return true;
}

}  // namespace matching

