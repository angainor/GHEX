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
#ifndef INCLUDED_GHEX_STRUCTURED_REMOTE_RANGE_HPP
#define INCLUDED_GHEX_STRUCTURED_REMOTE_RANGE_HPP

#include <cstring>
#include <vector>
#include <gridtools/common/host_device.hpp>

#include "../remote_range_traits.hpp"
#include "../transport_layer/ri/types.hpp"
#include "../arch_list.hpp"
#ifdef GHEX_USE_XPMEM
#include "../transport_layer/ri/xpmem/access_guard.hpp"
#include "../transport_layer/ri/xpmem/data.hpp"
#else
#include "../transport_layer/ri/thread/access_guard.hpp"
#endif /* GHEX_USE_XPMEM */

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

template<typename Field, typename Enable = void> // enable for gpu
struct field_view
{
    using layout = typename Field::layout_map;
    using dimension = typename Field::dimension;
    using value_type = typename Field::value_type;
    using coordinate = typename Field::coordinate_type;
    using strides_type = typename Field::strides_type;
#ifdef GHEX_USE_XPMEM
    using guard_type = tl::ri::xpmem::access_guard;
    using guard_view_type = tl::ri::xpmem::access_guard_view;
#else
    using guard_type = tl::ri::thread::access_guard;
    using guard_view_type = tl::ri::thread::access_guard_view;
#endif /* GHEX_USE_XPMEM */
    using rma_data_t = typename Field::rma_data_t;
    using size_type = tl::ri::size_type;

    Field m_field;
    rma_data_t m_rma_data;
    coordinate m_offset;
    coordinate m_extent;
    coordinate m_begin;
    coordinate m_end;
    coordinate m_reduced_stride;
    size_type  m_size;

    template<typename Array>
    field_view(const Field& f, const Array& offset, const Array& extent)
    : m_field(f)
    {
        static constexpr auto I = layout::find(dimension::value-1);
        m_size = 1;
        for (unsigned int i=0; i<dimension::value-1; ++i)
        {
            m_offset[i] = offset[i];
            m_extent[i] = extent[i];
            m_begin[i] = 0;
            m_end[i] = extent[i]-1;
            m_reduced_stride[i] = m_field.byte_strides()[i] / m_field.extents()[I];
            m_size *= extent[i];
        }
        if (Field::has_components::value)
        {
            unsigned int i = dimension::value-1;
            m_offset[i] = 0;
            m_extent[i] = f.num_components();
            m_begin[i] = 0;
            m_end[i] = m_extent[i]-1;
            m_reduced_stride[i] = m_field.byte_strides()[i] / m_field.extents()[I];
            m_size *= m_extent[i];
        }
        else
        {
            unsigned int i = dimension::value-1;
            m_offset[i] = offset[i];
            m_extent[i] = extent[i];
            m_begin[i] = 0;
            m_end[i] = extent[i]-1;
            m_reduced_stride[i] = m_field.byte_strides()[i] / m_field.extents()[I];
            m_size *= extent[i];
        }

        m_end[I] = extent[I];
        m_size /= extent[layout::find(dimension::value-1)];
    }

    field_view(const field_view&) = default;
    field_view(field_view&&) = default;
    
    GT_FUNCTION
    value_type& operator()(const coordinate& x) {
        return m_field(x+m_offset);
    }
    GT_FUNCTION
    const value_type& operator()(const coordinate& x) const {
        return m_field(x+m_offset);
    }
    
    GT_FUNCTION
    value_type* ptr(const coordinate& x) {
        return m_field.ptr(x+m_offset);
    }
    GT_FUNCTION
    const value_type* ptr(const coordinate& x) const {
        return m_field.ptr(x+m_offset);
    }
};

namespace detail {
template<unsigned int Dim, unsigned int D, typename Layout>
struct inc_coord
{
    template<typename Coord>
    static inline void fn(Coord& coord, const Coord& ext) noexcept
    {
        static constexpr auto I = Layout::find(Dim-D-1);
        if (coord[I] == ext[I] - 1)
        {
            coord[I] = 0;
            inc_coord<Dim, D + 1, Layout>::fn(coord, ext);
        }
        else
            coord[I] += 1;
    }
};
template<typename Layout>
struct inc_coord<3,1,Layout>
{
    template<typename Coord>
    static inline void fn(Coord& coord, const Coord& ext) noexcept
    {
        static constexpr auto Y = Layout::find(1);
        static constexpr auto Z = Layout::find(0);
        const bool cond = coord[Y] < ext[Y] - 1;
        coord[Y]  = cond ? coord[Y] + 1 : 0;
        coord[Z] += cond ? 0 : 1;
    }
};
template<unsigned int Dim, typename Layout>
struct inc_coord<Dim, Dim, Layout>
{
    template<typename Coord>
    static inline void fn(Coord& coord, const Coord& ext) noexcept
    {
        static constexpr auto I = Layout::find(Dim-1);
        for (unsigned int i = 0; i < Dim; ++i) coord[i] = ext[i] - 1;
        coord[I] = ext[I];
    }
};
} // namespace detail

template<typename Field, typename Enable = void> // enable for gpu
struct remote_range
{
    using view_type = field_view<Field>;
    using layout = typename Field::layout_map;
    using dimension = typename Field::dimension;
    using value_type = typename Field::value_type;
    using coordinate = typename Field::coordinate_type;
    using strides_type = typename Field::strides_type;
    using guard_type = typename view_type::guard_type;
    using guard_view_type = typename view_type::guard_view_type;
    using size_type = tl::ri::size_type;
    using iterator = range_iterator<remote_range>;

    guard_view_type   m_guard;
    view_type         m_view;
    size_type         m_chunk_size;
    
    remote_range(const view_type& v, guard_type& g) noexcept
    : m_guard{g}
    , m_view{v}
    , m_chunk_size{(size_type)(m_view.m_extent[layout::find(dimension::value-1)] * sizeof(value_type))}
    {}
    
    remote_range(const remote_range&) = default;
    remote_range(remote_range&&) = default;

    iterator  begin() const { return {const_cast<remote_range*>(this), 0, m_view.m_begin}; }
    iterator  end()   const { return {const_cast<remote_range*>(this), m_view.m_size, m_view.m_end}; }
    size_type buffer_size() const { return m_chunk_size; }

    // these functions are called at the remote site upon deserializing and reconstructing the range
    // and can be used to allocate state
    void init(tl::ri::remote_host_)   
    {
        m_view.m_field.reset_rma_data();
        m_view.m_field.init_rma_remote(m_view.m_rma_data);
	    m_guard.init_remote();
    }
    void init(tl::ri::remote_device_)
    {
        m_view.m_field.reset_rma_data();
        m_view.m_field.init_rma_remote(m_view.m_rma_data);
	    m_guard.init_remote();
    }
    void exit(tl::ri::remote_host_)
    {
        m_view.m_field.release_rma_remote();
        m_guard.release_remote(); 
    }
    void exit(tl::ri::remote_device_)
    {
        m_view.m_field.release_rma_remote();
        m_guard.release_remote(); 
    }
    
    static inline void put(const tl::ri::chunk& c, const tl::ri::byte* ptr, tl::ri::remote_host_, std::true_type)
    {
        // host to host put
        std::memcpy(c.data(), ptr, c.size());
    }
    static inline void put(const tl::ri::chunk& c, const tl::ri::byte* ptr, tl::ri::remote_host_, std::false_type)
    {
#ifdef __CUDACC__
        // devide to host put
        cudaMemcpy(c.data(), ptr, c.size(), cudaMemcpyHostToDevice);
#endif /* __CUDACC__ */
    }
    static inline void put(const tl::ri::chunk& c, const tl::ri::byte* ptr, tl::ri::remote_device_, std::true_type)
    {
#ifdef __CUDACC__
        cudaMemcpy(c.data(), ptr, c.size(), cudaMemcpyDeviceToHost);
#endif /* __CUDACC__ */
    }
    static inline void put(const tl::ri::chunk& c, const tl::ri::byte* ptr, tl::ri::remote_device_, std::false_type)
    {
#ifdef __CUDACC__
        // devide to host put
        cudaMemcpy(c.data(), ptr, c.size(), cudaMemcpyDeviceToDevice);
#endif /* __CUDACC__ */
    }
    
    static inline void put(const tl::ri::chunk& c, const tl::ri::byte* ptr, tl::ri::remote_host_ h) {
        put(c, ptr, h, std::is_same<typename Field::arch_type, cpu>{});
    }
    static inline void put(const tl::ri::chunk& c, const tl::ri::byte* ptr, tl::ri::remote_device_ h) {
        put(c, ptr, h, std::is_same<typename Field::arch_type, cpu>{});
    }
    
    void start_local_epoch() { m_guard.start_local_epoch(); }
    void end_local_epoch()   { m_guard.end_local_epoch(); }

    void start_remote_epoch(tl::ri::remote_host_)   { m_guard.start_remote_epoch(); }
    void end_remote_epoch(tl::ri::remote_host_)     { m_guard.end_remote_epoch(); }
    void start_remote_epoch(tl::ri::remote_device_) { m_guard.start_remote_epoch(); }
    void end_remote_epoch(tl::ri::remote_device_)   { m_guard.end_remote_epoch(); }
    
    tl::ri::chunk get_chunk(const coordinate& coord) const noexcept {
        auto ptr = const_cast<tl::ri::byte*>(reinterpret_cast<const tl::ri::byte*>(&(m_view(coord))));
        return {ptr, m_chunk_size};
    }

    size_type inc(size_type index, size_type n, coordinate& coord) const noexcept {
        if (n < 0 && -n > index)
        {
            coord = m_view.m_begin;
            return 0;
        }
        index += n;
        if (index >= m_view.m_size)
        {
            coord = m_view.m_end;
            return m_view.m_size;
        }
        else
        {
            auto idx = index;
            static constexpr auto I = layout::find(dimension::value-1);
            coord[I] = 0;
            for (unsigned int d = 0; d < dimension::value; ++d)
            {
                const auto i = layout::find(d);
                coord[i] = index / m_view.m_reduced_stride[i];
                index -= coord[i] * m_view.m_reduced_stride[i];
            }
            return idx;
        }
    }

    size_type inc(size_type index, coordinate& coord) const noexcept {
        static constexpr auto I = layout::find(dimension::value-1);
        if (index + 1 >= m_view.m_size)
        {
            coord = m_view.m_end;
            return m_view.m_size;
        }
        else
        {
            coord[I] = 0;
            detail::inc_coord<dimension::value, 1, layout>::fn(coord, m_view.m_extent);
            return index + 1;
        }
    }
};

template<typename Field>
struct remote_range_generator
{
    using range_type = remote_range<Field>;
    using guard_type = typename range_type::guard_type;
    using guard_view_type = typename range_type::guard_view_type;

    template<typename RangeFactory, typename Communicator>
    struct target_range
    {
        using rank_type = typename Communicator::rank_type;
        using tag_type = typename Communicator::tag_type;

        Communicator m_comm;
        guard_type m_guard;
        field_view<Field> m_view;
        typename RangeFactory::range_type m_local_range;
        rank_type m_dst;
        tag_type m_tag;
        typename Communicator::template future<void> m_request;
        std::vector<tl::ri::byte> m_archive;

        template<typename Coord>
        target_range(const Communicator& comm, const Field& f, const Coord& first, const Coord& last, rank_type dst, tag_type tag)
        : m_comm{comm}
        , m_guard{}
        , m_view{f, first, last-first+1}
        , m_dst{dst}
        , m_tag{tag}
        {
            m_archive.resize(RangeFactory::serial_size);
            m_view.m_field.init_rma_local();
            m_view.m_rma_data = m_view.m_field.get_rma_data();
	        m_local_range = {RangeFactory::template create<range_type>(m_view,m_guard)};
            RangeFactory::serialize(m_local_range, m_archive.data());
            m_request = m_comm.send(m_archive, m_dst, m_tag);
        }

        void send()
        {
            //m_comm.send(m_archive, m_dst, m_tag).wait();
            m_request.wait();
        }

        void release()
        {
            m_view.m_field.release_rma_local();
        }
    };

    template<typename RangeFactory, typename Communicator>
    struct source_range
    {
        using rank_type = typename Communicator::rank_type;
        using tag_type = typename Communicator::tag_type;

        Communicator m_comm;
        field_view<Field> m_view;
        typename RangeFactory::range_type m_remote_range;
        rank_type m_src;
        tag_type m_tag;
        typename Communicator::template future<void> m_request;
        std::vector<tl::ri::byte> m_buffer;
        typename RangeFactory::range_type::iterator_type m_pos;

        template<typename Coord>
        source_range(const Communicator& comm, const Field& f, const Coord& first, const Coord& last, rank_type src, tag_type tag)
        : m_comm{comm}
        , m_view{f, first, last-first+1}
        , m_src{src}
        , m_tag{tag}
        {
            m_buffer.resize(RangeFactory::serial_size);
            m_request = m_comm.recv(m_buffer, m_src, m_tag);
        }

        void recv()
        {
            m_request.wait();
            m_remote_range = RangeFactory::deserialize(tl::ri::host, m_buffer.data());
            m_buffer.resize(m_remote_range.buffer_size());
            m_pos = m_remote_range.begin();
        }
        
        void release()
        {
        }
    };
};

} // namespace structured

template<>
struct remote_range_traits<structured::remote_range_generator>
{
    template<typename Communicator>
    static tl::ri::locality is_local(Communicator comm, int remote_rank)
    {
        if (comm.rank() == remote_rank) return tl::ri::locality::thread;
#ifdef GHEX_USE_XPMEM
        else if (comm.is_local(remote_rank)) return tl::ri::locality::process;
#endif /* GHEX_USE_XPMEM */
        else return tl::ri::locality::remote;
    }
};

} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_REMOTE_RANGE_HPP */
