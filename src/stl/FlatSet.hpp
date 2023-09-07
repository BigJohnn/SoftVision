// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <set>

namespace stl
{
//  template <class Key
//         ,class Compare  = std::less<Key>
//         ,class Allocator = std::allocator<Key> >
//  using flat_set = boost::container::flat_set<Key, Compare, Allocator>;

    //TODO: use some real flat_set!
    template <class _Key, class _Compare = std::less<_Key>,
          class _Allocator = std::allocator<_Key> >
    using flat_set = std::set<_Key, _Compare, _Allocator>;
}
