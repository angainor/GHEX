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
#ifndef INCLUDED_GHEX_STRUCTURED_RMA_RANGE_ITERATOR_HPP
#define INCLUDED_GHEX_STRUCTURED_RMA_RANGE_ITERATOR_HPP

#include "../transport_layer/ri/types.hpp"

namespace gridtools {
namespace ghex {
namespace structured {

template<typename Range>
struct range_iterator
{
    using coordinate = typename Range::coordinate;
    using chunk = tl::ri::chunk;
    using size_type = tl::ri::size_type;

    Range*      m_range;
    size_type   m_index;
    coordinate  m_coord;

    range_iterator(Range* r, size_type idx, const coordinate& coord)
    : m_range{r}
    , m_index{idx}
    , m_coord{coord}
    {}
    range_iterator(const range_iterator&) = default;
    range_iterator(range_iterator&&) = default;

    chunk     operator*() const noexcept { return m_range->get_chunk(m_coord); }
    void      operator++() noexcept { m_index = m_range->inc(m_index, m_coord); }
    void      operator--() noexcept { m_index = m_range->inc(m_index, -1, m_coord); }
    void      operator+=(size_type n) noexcept { m_index = m_range->inc(m_index, n, m_coord); }
    size_type sub(const range_iterator& other) const { return m_index - other.m_index; }
    bool      equal(const range_iterator& other) const { return m_index == other.m_index; }
    bool      lt(const range_iterator& other) const { return m_index < other.m_index; }
};

} // namespace structured
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_RMA_RANGE_ITERATOR_HPP */