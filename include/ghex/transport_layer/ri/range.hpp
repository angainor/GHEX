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
#ifndef INCLUDED_GHEX_TRANSPORT_LAYER_RI_RANGE_HPP
#define INCLUDED_GHEX_TRANSPORT_LAYER_RI_RANGE_HPP

#include "./range_iface.hpp"
#include "./iterator.hpp"

namespace gridtools {
namespace ghex {
namespace tl {
namespace ri {

template<unsigned int StackMemory, unsigned int IteratorStackMemory>
struct put_range
{
    using iterator_type = iterator<IteratorStackMemory>;

    int  m_id = 0;
    byte m_stack[StackMemory];

    template<typename Range, typename Arch>
    put_range(Range&& r, Arch, int id = 0) {
        using range_t = std::remove_cv_t<std::remove_reference_t<Range>>;
        new (m_stack) put_range_impl<range_t, iterator_type, Arch>{std::forward<Range>(r)};
        m_id = id;
    }
    put_range() = default;
    put_range(put_range&&) = default;
    put_range& operator=(put_range&&) = default;

    iterator_type begin() const noexcept { return ciface().begin(); }
    iterator_type end() const noexcept { return ciface().end(); }
    chunk operator[](size_type i) const noexcept { return *(begin() + i); }
    size_type size() const noexcept { return end() - begin(); }

    size_type buffer_size() const { return ciface().buffer_size(); }

    void start_target_epoch() { iface().start_local_epoch(); }
    void end_target_epoch()   { iface().end_local_epoch(); }
    void start_source_epoch() { iface().start_remote_epoch(); }
    void end_source_epoch()   { iface().end_remote_epoch(); }

    put_range_iface<iterator_type>&  iface()
    {
        return *reinterpret_cast<put_range_iface<iterator_type>*>(m_stack);
    }
    const put_range_iface<iterator_type>&  iface() const
    {
        return *reinterpret_cast<const put_range_iface<iterator_type>*>(m_stack);
    }
    const put_range_iface<iterator_type>& ciface() const
    {
        return *reinterpret_cast<const put_range_iface<iterator_type>*>(m_stack);
    }
};

} // namespace ri
} // namespace tl
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TRANSPORT_LAYER_RI_RANGE_HPP */