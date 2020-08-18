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
#ifndef INCLUDED_GHEX_TRANSPORT_LAYER_RI_ITERATOR_HPP
#define INCLUDED_GHEX_TRANSPORT_LAYER_RI_ITERATOR_HPP

#include "./iterator_iface.hpp"

namespace gridtools {
namespace ghex {
namespace tl {
namespace ri {

template<unsigned int StackMemory>
struct iterator
{
    byte m_stack[StackMemory];

    iterator() noexcept {}

    template<typename Iterator>
    iterator(const Iterator& it) noexcept {
        new (m_stack) iterator_impl<Iterator>{it};
    }

    //template<typename Iterator>
    iterator(const byte* buffer) noexcept {
        std::memcpy(m_stack, buffer, StackMemory);
    }

    iterator(const iterator&) = default;
    iterator(iterator&&) = default;
    iterator& operator=(const iterator&) = default;
    iterator& operator=(iterator&&) = default;

    chunk operator*() const noexcept { return *ciface(); }

    arrow_proxy<chunk> operator->() const noexcept { return {*ciface()}; }

    operator chunk() const noexcept { return (chunk)ciface(); }

    iterator& operator++()            noexcept { ++iface(); return *this; }
    iterator  operator++(int)         noexcept { const iterator tmp(*this); ++iface(); return tmp; }
    iterator& operator--()            noexcept { --iface(); return *this; }
    iterator  operator--(int)         noexcept { const iterator tmp(*this); --iface(); return tmp; }
    iterator& operator+=(size_type n) noexcept { iface() += n; return *this; }
    iterator& operator-=(size_type n) noexcept { iface() += -n; return *this; }

    friend inline iterator operator+(iterator a, size_type n) noexcept { return a += n; }
    friend inline iterator operator-(iterator a, size_type n) noexcept { return a -= n; }

    friend inline size_type operator-(const iterator& a, const iterator& b) { return a.ciface().sub(b.ciface()); }

    friend inline bool operator==(const iterator& a, const iterator& b) noexcept { return a.ciface().equal(b.ciface()); }
    friend inline bool operator!=(const iterator& a, const iterator& b) noexcept { return !(a == b); }
    friend inline bool operator<(const iterator& a, const iterator& b)  noexcept { return a.ciface().lt(b.ciface()); }
    friend inline bool operator<=(const iterator& a, const iterator& b) noexcept { return (a < b) || (a == b); }
    friend inline bool operator>(const iterator& a, const iterator& b)  noexcept { return !(a <= b); }
    friend inline bool operator>=(const iterator& a, const iterator& b) noexcept { return !(a < b); }

          iterator_iface&  iface()       { return *reinterpret_cast<iterator_iface*>(m_stack); }
    const iterator_iface&  iface() const { return *reinterpret_cast<const iterator_iface*>(m_stack); }
    const iterator_iface& ciface() const { return *reinterpret_cast<const iterator_iface*>(m_stack); }
};

} // namespace ri
} // namespace tl
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TRANSPORT_LAYER_RI_ITERATOR_HPP */

