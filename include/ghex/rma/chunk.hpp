/*
 * GridTools
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */
#ifndef INCLUDED_GHEX_RMA_CHUNK_HPP
#define INCLUDED_GHEX_RMA_CHUNK_HPP

#include <gridtools/common/host_device.hpp>

namespace gridtools {
namespace ghex {
namespace rma {

// chunk of contiguous memory of type T
// used in range iterators for example
template<typename T>
struct chunk
{
    using value_type = T;
    using size_type = unsigned long;
    
    T* m_ptr;
    size_type m_size;

    GT_FUNCTION
    T* data() const noexcept { return m_ptr; }
    GT_FUNCTION
    T operator[](unsigned int i) const noexcept { return m_ptr[i]; }
    GT_FUNCTION
    T& operator[](unsigned int i) noexcept { return m_ptr[i]; }
    GT_FUNCTION
    size_type size() const noexcept { return m_size; }
    GT_FUNCTION
    size_type bytes() const noexcept { return m_size*sizeof(T); }
};

} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_CHUNK_HPP */
