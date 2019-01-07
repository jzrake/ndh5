#pragma once
#include <string>
#include <vector>
#include <hdf5.h>




// ============================================================================
namespace h5
{
    template <class GroupType, class DatasetType>
    class Location;
    class Link;
    class File;
    class Group;
    class Dataset;
    class Datatype;
    class Dataspace;

    enum class Intent { rdwr, rdonly, swmr_write, swmr_read };
    enum class Object { file, group, dataset };

    namespace detail {
        class hyperslab;
        static inline herr_t get_last_error(unsigned, const H5E_error2_t*, void*);
        template<typename T> static inline T check(T);
        template<typename T> static inline Datatype make_datatype_for(const T&);
        template<typename T> static inline Datatype make_datatype_for(const std::vector<T>&);
        template<typename T> static inline Dataspace make_dataspace_for(const T&, bool selected_part=false);
        template<typename T> static inline Dataspace make_dataspace_for(const std::vector<T>&, bool selected_part=false);
        template<typename T> static inline void prepare(const Datatype&, const Dataspace&, T&);
        template<typename T> static inline void prepare(const Datatype&, const Dataspace&, std::vector<T>&);
        template<typename T> static inline void* get_address(T&);
        template<typename T> static inline void* get_address(std::vector<T>&);
        template<typename T> static inline const void* get_address(const T&);
        template<typename T> static inline const void* get_address(const std::vector<T>&);

        template<typename T, int R> static inline Datatype make_datatype_for(const nd::ndarray<T, R>&);
        template<typename T, int R> static inline Dataspace make_dataspace_for(const nd::ndarray<T, R>&, bool selected_part=false);
        template<typename T, int R> static inline void prepare(const Datatype&, const Dataspace&, nd::ndarray<T, R>&);
        template<typename T, int R> static inline void* get_address(nd::ndarray<T, R>&);
        template<typename T, int R> static inline const void* get_address(const nd::ndarray<T, R>&);
    }
}




// ============================================================================
struct h5::detail::hyperslab
{
    hyperslab() {}

    template<typename Selector>
    hyperslab(Selector sel)
    {
        auto sel_shape = sel.shape();
        start = std::vector<hsize_t>(sel.start.begin(), sel.start.end());
        skips = std::vector<hsize_t>(sel.skips.begin(), sel.skips.end());
        count = std::vector<hsize_t>(sel_shape.begin(), sel_shape.end());
        block = std::vector<hsize_t>(sel.rank, 1);
    }

    void check_valid(hsize_t rank) const
    {
        if (start.size() != rank ||
            skips.size() != rank ||
            count.size() != rank ||
            block.size() != rank)
        {
            throw std::invalid_argument("inconsistent selection sizes");
        }
    }

    void select(hid_t space_id)
    {
        check_valid(detail::check(H5Sget_simple_extent_ndims(space_id)));
        detail::check(H5Sselect_hyperslab(space_id, H5S_SELECT_SET,
            start.data(),
            skips.data(),
            count.data(),
            block.data()));
    }

    std::vector<hsize_t> start;
    std::vector<hsize_t> count;
    std::vector<hsize_t> skips;
    std::vector<hsize_t> block;
};

herr_t h5::detail::get_last_error(unsigned n, const H5E_error2_t *err, void *data)
{
    if (n == 0)
    {
        *static_cast<H5E_error2_t*>(data) = *err;
    }
    return 0;
}

template<typename T>
T h5::detail::check(T result)
{
    if (result < 0)
    {
        H5E_error2_t err;
        hid_t eid = H5Eget_current_stack();
        H5Ewalk(eid, H5E_WALK_UPWARD, get_last_error, &err);
        std::string what = err.desc;
        H5Eclear(eid);
        H5Eclose_stack(eid);
        throw std::invalid_argument(what);
    }
    return result;
}

template<>
inline void* h5::detail::get_address<std::string>(std::string& val)
{
    return &val[0];
}

template<typename T>
inline void* h5::detail::get_address(std::vector<T>& val)
{
    return &val[0];
}

template<typename T>
inline void* h5::detail::get_address(T& val)
{
    return &val;
}

template<typename T, int R>
inline void* h5::detail::get_address(nd::ndarray<T, R>& val)
{
    return val.data();
}

template<>
const inline void* h5::detail::get_address<std::string>(const std::string& val)
{
    return &val[0];
}

template<typename T>
const inline void* h5::detail::get_address(const std::vector<T>& val)
{
    return &val[0];
}

template<typename T>
const inline void* h5::detail::get_address(const T& val)
{
    return &val;
}

template<typename T, int R>
const inline void* h5::detail::get_address(const nd::ndarray<T, R>& val)
{
    return val.data();
}




// ============================================================================
class h5::Datatype final
{
public:
    Datatype() {}

    Datatype(const Datatype& other)
    {
        id = H5Tcopy(other.id);
    }

    Datatype(Datatype&& other)
    {
        id = other.id;
        other.id = -1;
    }

    ~Datatype()
    {
        close();
    }

    Datatype& operator=(const Datatype& other)
    {
        close();
        id = H5Tcopy(other.id);
        return *this;
    }

    bool operator==(const Datatype& other) const
    {
        return detail::check(H5Tequal(id, other.id));
    }

    bool operator!=(const Datatype& other) const
    {
        return ! operator==(other);
    }

    void close()
    {
        if (id != -1)
        {
            H5Tclose(id);
        }
    }

    std::size_t size() const
    {
        return detail::check(H5Tget_size(id));
    }

    Datatype with_size(std::size_t size) const
    {
        Datatype other = *this;
        H5Tset_size(other.id, size);
        return other;
    }

private:
    // ========================================================================
    template<typename T> friend Datatype detail::make_datatype_for(const T&);
    template<typename T> friend Datatype detail::make_datatype_for(const std::vector<T>&);
    template<typename T, int R> friend Datatype detail::make_datatype_for(const nd::ndarray<T, R>&); // NEED?
    friend class Link;
    friend class Dataset;

    Datatype(hid_t id) : id(id) {}
    hid_t id = -1;
};




// ============================================================================
template<>
inline h5::Datatype h5::detail::make_datatype_for<std::string>(const std::string& val)
{
    return Datatype(H5Tcopy(H5T_C_S1)).with_size(val.size());
}

template<>
inline h5::Datatype h5::detail::make_datatype_for<char>(const char&)
{
    return H5Tcopy(H5T_C_S1);
}

template<>
inline h5::Datatype h5::detail::make_datatype_for<int>(const int&)
{
    return H5Tcopy(H5T_NATIVE_INT);
}

template<>
inline h5::Datatype h5::detail::make_datatype_for<double>(const double&)
{
    return H5Tcopy(H5T_NATIVE_DOUBLE);
}

template<typename T>
inline h5::Datatype h5::detail::make_datatype_for(const std::vector<T>&)
{
    return make_datatype_for(T());
}

template<typename T, int R>
inline h5::Datatype h5::detail::make_datatype_for(const nd::ndarray<T, R>&)
{
    return make_datatype_for(T());
}




// ============================================================================
class h5::Dataspace
{
public:
    static Dataspace scalar()
    {
        return detail::check(H5Screate(H5S_SCALAR));
    }

    template<typename Container>
    static Dataspace simple(Container dims)
    {
        auto hdims = std::vector<hsize_t>(dims.begin(), dims.end());
        return detail::check(H5Screate_simple(int(hdims.size()), &hdims[0], nullptr));
    }

    Dataspace()
    {
        id = detail::check(H5Screate(H5S_NULL));
    }

    template<int Rank, int Axis>
    Dataspace(nd::selector<Rank, Axis> sel)
    {
        auto dims = std::vector<hsize_t>(sel.count.begin(), sel.count.end());
        auto slab = detail::hyperslab(sel);
        id = detail::check(H5Screate_simple(int(dims.size()), &dims[0], nullptr));
        slab.select(id);
    }

    Dataspace(const Dataspace& other)
    {
        id = detail::check(H5Scopy(other.id));
    }

    Dataspace(std::initializer_list<std::size_t> dims)
    {
        *this = dims.size() == 0 ? scalar() : simple(std::vector<size_t>(dims));
    }

    ~Dataspace()
    {
        close();
    }

    Dataspace& operator=(const Dataspace& other)
    {
        close();
        id = detail::check(H5Scopy(other.id));
        return *this;
    }

    bool operator==(const Dataspace& other) const
    {
        return detail::check(H5Sextent_equal(id, other.id));
    }

    bool operator!=(const Dataspace& other) const
    {
        return ! operator==(other);
    }

    void close()
    {
        if (id != -1)
        {
            detail::check(H5Sclose(id));
        }
    }

    std::size_t rank() const
    {
        return detail::check(H5Sget_simple_extent_ndims(id));
    }

    std::size_t size() const
    {
        return detail::check(H5Sget_simple_extent_npoints(id));
    }

    std::vector<std::size_t> extent() const
    {
        auto ext = std::vector<hsize_t>(rank());
        detail::check(H5Sget_simple_extent_dims(id, &ext[0], nullptr));
        return std::vector<std::size_t>(ext.begin(), ext.end());
    }

    std::size_t selection_size() const
    {
        return detail::check(H5Sget_select_npoints(id));
    }

    std::vector<std::size_t> selection_lower() const
    {
        auto lower = std::vector<hsize_t>(rank());
        auto upper = std::vector<hsize_t>(rank());
        detail::check(H5Sget_select_bounds(id, &lower[0], &upper[0]));
        return std::vector<std::size_t>(lower.begin(), lower.end());
    }

    std::vector<std::size_t> selection_upper() const
    {
        auto lower = std::vector<hsize_t>(rank());
        auto upper = std::vector<hsize_t>(rank());
        detail::check(H5Sget_select_bounds(id, &lower[0], &upper[0]));
        return std::vector<std::size_t>(upper.begin(), upper.end());
    }

    Dataspace& select_all()
    {
        detail::check(H5Sselect_all(id));
        return *this;
    }

    Dataspace& select_none()
    {
        detail::check(H5Sselect_none(id));
        return *this;
    }

private:
    // ========================================================================
    friend class Link;
    friend class Dataset;

    Dataspace(hid_t id) : id(id) {}
    hid_t id = -1;
};




// ============================================================================
template<typename T>
h5::Dataspace h5::detail::make_dataspace_for(const T&, bool)
{
    return Dataspace::scalar();
}

template<typename T>
h5::Dataspace h5::detail::make_dataspace_for(const std::vector<T>& val, bool)
{
    return Dataspace{val.size()};
}

template<typename T, int R>
h5::Dataspace h5::detail::make_dataspace_for(const nd::ndarray<T, R>& val, bool selected_part)
{
    if (selected_part)
    {
        return Dataspace::simple(val.shape());
    }
    return val.get_selector();
}




// ============================================================================
template<typename T>
inline void h5::detail::prepare(const Datatype&, const Dataspace&, T&)
{
}

template<>
inline void h5::detail::prepare(const Datatype& type, const Dataspace&, std::string& value)
{
    value.resize(type.size());
}

template<typename T>
inline void h5::detail::prepare(const Datatype&, const Dataspace& space, std::vector<T>& value)
{
    value.resize(space.selection_size());
}

template<typename T, int R>
inline void h5::detail::prepare(const Datatype&, const Dataspace& space, nd::ndarray<T, R>& val)
{
    std::array<int, R> dim_sizes;

    if (space.rank() != R)
    {
        throw std::invalid_argument("data space has the wrong rank for target array");
    }

    auto upper = space.selection_upper();
    auto lower = space.selection_lower();

    for (int n = 0; n < R; ++n)
    {
        dim_sizes[n] = int(upper[n]) - int(lower[n]) + 1;
    }

    auto target = nd::array<T, R>(dim_sizes);
    val.become(target);
}




// ============================================================================
class h5::Link
{
private:

    Link() {}

    Link(hid_t id) : id(id) {}

    Link(Link&& other)
    {
        id = other.id;
        other.id = -1;
    }

    Link(const Link& other) = delete;

    Link& operator=(Link&& other)
    {
        id = other.id;
        other.id = -1;
        return *this;
    }

    void close(Object object)
    {
        if (id != -1)
        {
            switch (object)
            {
                case Object::file   : H5Fclose(id); break;
                case Object::group  : H5Gclose(id); break;
                case Object::dataset: H5Dclose(id); break;
            }
            id = -1;
        }
    }

    std::size_t size() const
    {
        auto op = [] (auto, auto, auto, auto) { return 0; };
        auto idx = hsize_t(0);
        H5Literate(id, H5_INDEX_NAME, H5_ITER_INC, &idx, op, nullptr);
        return idx;
    }

    bool contains(const std::string& name, Object object) const
    {
        if (H5Lexists(id, name.data(), H5P_DEFAULT))
        {
            H5O_info_t info;
            H5Oget_info_by_name(id, name.data(), &info, H5P_DEFAULT);

            switch (object)
            {
                case Object::file   : return false;
                case Object::group  : return info.type == H5O_TYPE_GROUP;
                case Object::dataset: return info.type == H5O_TYPE_DATASET;
            }
        }
        return false;
    }

    Link open_group(const std::string& name)
    {
        return detail::check(H5Gopen(id, name.data(),
            H5P_DEFAULT));
    }

    Link create_group(const std::string& name)
    {
        return detail::check(H5Gcreate(id, name.data(),
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    }

    Link open_dataset(const std::string& name)
    {
        return detail::check(H5Dopen(id, name.data(),
            H5P_DEFAULT));
    }

    Link create_dataset(const std::string& name,
                        const Datatype& type,
                        const Dataspace& space)
    {
        return detail::check(H5Dcreate(
            id,
            name.data(),
            type.id,
            space.id,
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT));
    }




    // ========================================================================
    class iterator
    {
    public:
        using value_type = std::string;
        using iterator_category = std::forward_iterator_tag;
        iterator(hid_t id, hsize_t idx) : id(id), idx(idx) {}
        iterator& operator++() { ++idx; return *this; }
        iterator operator++(int) { auto ret = *this; this->operator++(); return ret; }
        bool operator==(iterator other) const { return id == other.id && idx == other.idx; }
        bool operator!=(iterator other) const { return id != other.id || idx != other.idx; }
        std::string operator*() const
        {
            static char name[1024];

            if (H5Lget_name_by_idx(id, ".",
                H5_INDEX_NAME, H5_ITER_NATIVE, idx, name, 1024, H5P_DEFAULT) > 1024)
            {
                throw std::overflow_error("object names longer than 1024 are not supported");
            }
            return name;
        }

    private:
        hid_t id = -1;
        hsize_t idx = 0;
    };

    iterator begin() const
    {
        return iterator(id, 0);
    }

    iterator end() const
    {
        return iterator(id, size());
    }




    // ========================================================================
    template <class GroupType, class DatasetType>
    friend class Location;
    friend class File;
    friend class Group;
    friend class Dataset;

    hid_t id = -1;
};




// ============================================================================
class h5::Dataset final
{
public:

    Dataset() {}

    Dataset(const Dataset&) = delete;

    Dataset(Dataset&& other)
    {
        link = std::move(other.link);
    }

    Dataset& operator=(Dataset&& other)
    {
        link = std::move(other.link);
        return *this;
    }

    ~Dataset()
    {
        close();
    }

    void close()
    {
        link.close(Object::dataset);
    }

    Dataspace get_space() const
    {
        return detail::check(H5Dget_space(link.id));
    }

    Datatype get_type() const
    {
        return detail::check(H5Dget_type(link.id));
    }

    template<typename T>
    void write(const T& value)
    {
        write(value, get_space());
    }

    template<typename T, typename Selector>
    T write(const T& value, Selector sel)
    {
        auto extent = get_space().extent();
        auto fspace = Dataspace(nd::with_count(sel, extent.begin(), extent.end()));
        return write(value, fspace);
    }

    template<typename T>
    void write(const T& value, const Dataspace& fspace)
    {
        auto data = detail::get_address(value);
        auto type = detail::make_datatype_for(value);
        auto mspace = detail::make_dataspace_for(value);
        check_compatible(type);
        detail::check(H5Dwrite(link.id, type.id, mspace.id, fspace.id, H5P_DEFAULT, data));
    }

    template<typename T>
    T read()
    {
        return read<T>(get_space());
    }

    template<typename T, typename Selector>
    T read(Selector sel)
    {
        auto extent = get_space().extent();
        auto fspace = Dataspace(nd::with_count(sel, extent.begin(), extent.end()));
        return read<T>(fspace);
    }

    template<typename T>
    T read(const Dataspace& fspace)
    {
        T value;
        detail::prepare(get_type(), fspace, value);
        auto data = detail::get_address(value);
        auto type = detail::make_datatype_for(value);
        auto mspace = detail::make_dataspace_for(value);
        check_compatible(type);
        detail::check(H5Dread(link.id, type.id, mspace.id, fspace.id, H5P_DEFAULT, data));
        return value;
    }

private:
    // ========================================================================
    Datatype check_compatible(const Datatype& type) const
    {
        if (type != get_type())
        {
            throw std::invalid_argument("source and target have different data types");
        }
        return type;
    }

    // ========================================================================
    template <class GroupType, class DatasetType>
    friend class Location;

    Dataset(Link link) : link(std::move(link)) {}
    Link link;
};




// ============================================================================
template <class GroupType, class DatasetType>
class h5::Location
{
public:

    Location() {}

    Location(const Group&) = delete;

    Location(Location&& other)
    {
        link = std::move(other.link);
    }

    Location& operator=(Location&& other)
    {
        link = std::move(other.link);
        return *this;
    }

    bool is_open() const
    {
        return link.id != -1;
    }

    std::size_t size() const
    {
        return link.size();
    }

    Link::iterator begin() const
    {
        return link.begin();
    }

    Link::iterator end() const
    {
        return link.end();
    }

    GroupType operator[](const std::string& name)
    {
        return require_group(name);
    }

    GroupType open_group(const std::string& name)
    {
        return link.open_group(name);
    }

    GroupType require_group(const std::string& name)
    {
        if (link.contains(name, Object::group))
        {
            return open_group(name);
        }
        return link.create_group(name);
    }

    DatasetType open_dataset(const std::string& name)
    {
        return link.open_dataset(name);
    }

    DatasetType require_dataset(const std::string& name,
                                const Datatype& type,
                                const Dataspace& space)
    {
        if (link.contains(name, Object::dataset))
        {
            auto dset = open_dataset(name);

            if (dset.get_type() == type &&
                dset.get_space() == space)
            {
                return dset;
            }
            throw std::invalid_argument(
                "data set with different type or space already exists");
        }
        return link.create_dataset(name, type, space);
    }

    template<typename T>
    DatasetType require_dataset(const std::string& name, const Dataspace& space={})
    {
        return require_dataset(name, detail::make_datatype_for(T()), space);
    }

    template<typename T>
    void write(const std::string& name, const T& value)
    {
        auto type = detail::make_datatype_for(value);
        auto space = detail::make_dataspace_for(value, true);
        require_dataset(name, type, space).write(value);
    }

    template<typename T, typename Selector>
    void write(const std::string& name, const T& value, Selector sel)
    {
        auto type = detail::make_datatype_for(value);
        auto space = detail::make_dataspace_for(value, true);
        require_dataset(name, type, space).write(value, sel);
    }

    template<typename T>
    T read(const std::string& name)
    {
        return open_dataset(name).template read<T>();
    }

    template<typename T, typename Selector>
    T read(const std::string& name, Selector sel)
    {
        return open_dataset(name).template read<T>(sel);
    }

protected:
    // ========================================================================
    Location(Link link) : link(std::move(link)) {}
    Link link;
};




// ============================================================================
class h5::Group final : public Location<Group, Dataset>
{
public:
    Group() {}

    Group(const Group&) = delete;

    Group(Group&& other)
    {
        link = std::move(other.link);
    }

    ~Group()
    {
        close();
    }

    Group& operator=(Group&& other)
    {
        link = std::move(other.link);
        return *this;
    }

    void close()
    {
        link.close(Object::group);
    }

private:
    template <class GroupType, class DatasetType> friend class h5::Location;
    Group(Link link) : Location(std::move(link)) {}
};




// ============================================================================
class h5::File final : public Location<Group, Dataset>
{
public:

    static bool exists(const std::string& filename)
    {
        return H5Fis_hdf5(filename.data()) > 0;
    }

    File() {}

    File(const std::string& filename, const std::string& mode="r")
    {
        if (mode == "r")
        {
            link.id = detail::check(H5Fopen(filename.data(), H5F_ACC_RDONLY, H5P_DEFAULT));
        }
        else if (mode == "r+")
        {
            link.id = detail::check(H5Fopen(filename.data(), H5F_ACC_RDWR, H5P_DEFAULT));
        }
        else if (mode == "w")
        {
            link.id = detail::check(H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
        }
        else
        {
            throw std::invalid_argument("File mode must be r, r+, or w");
        }
    }

    File(const File&) = delete;

    File(File&& other)
    {
        link = std::move(other.link);
    }

    ~File()
    {
        close();
    }

    File& operator=(File&& other)
    {
        link = std::move(other.link);
        return *this;
    }

    void close()
    {
        link.close(Object::file);
    }

    Intent intent() const
    {
        unsigned intent;
        detail::check(H5Fget_intent(link.id, &intent));

        if (intent == H5F_ACC_RDWR)       return Intent::rdwr;
        if (intent == H5F_ACC_RDONLY)     return Intent::rdonly;
        // if (intent == H5F_ACC_SWMR_WRITE) return Intent::swmr_write;
        // if (intent == H5F_ACC_SWMR_READ)  return Intent::swmr_read;

        throw;
    }

private:
    // ========================================================================
    File(Link link) : Location(std::move(link)) {}
};




// ============================================================================
#ifdef TEST_NDH5
#include "catch.hpp"
#include <array>




SCENARIO("Files can be created", "[h5::File]")
{
    GIVEN("A file opened for writing")
    {
        auto file = h5::File("test.h5", "w");

        THEN("The file reports being opened with read/write intent")
        {
            REQUIRE(file.is_open());
            REQUIRE(file.intent() == h5::Intent::rdwr);
        }

        WHEN("The file is closed manually")
        {
            file.close();

            THEN("It reports as not open, and can be closed again without effect")
            {
                REQUIRE_FALSE(file.is_open());
                REQUIRE_NOTHROW(file.close());
            }
        }
    }

    GIVEN("A file is opened for reading")
    {
        auto file = h5::File("test.h5", "r");
    
        THEN("It reports as open with read-only intent")
        {
            REQUIRE(file.is_open());
            REQUIRE(file.intent() == h5::Intent::rdonly);
        }
    }

    GIVEN("A filename that does not exist")
    {
        THEN("h5::File::exists reports it does not exist, and open as read throws")
        {
            REQUIRE_FALSE(h5::File::exists("no-exist.h5"));
            REQUIRE_THROWS(h5::File("no-exist.h5", "r"));
        }
    }
}


SCENARIO("Groups can be created in files", "[h5::Group]")
{
    GIVEN("A file opened for writing")
    {
        auto file = h5::File("test.h5", "w");

        WHEN("Three groups are created")
        {
            auto group1 = file.require_group("group1");
            auto group2 = file.require_group("group2");
            auto group3 = file.require_group("group3");

            THEN("file.size() returns 3")
            {
                REQUIRE(file.size() == 3);
            }

            THEN("The groups can be opened without throwing, but a non-existent group does throw")
            {
                REQUIRE_NOTHROW(file.open_group("group1"));
                REQUIRE_NOTHROW(file.open_group("group2"));
                REQUIRE_NOTHROW(file.open_group("group3"));
                REQUIRE_THROWS(file.open_group("no-exist"));

                REQUIRE_NOTHROW(file["group1"]);
                REQUIRE_NOTHROW(file["group1"]["new-group"]); // creates a new group
            }

            THEN("The groups have the correct names")
            {
                int n = 0;

                for (auto group : file)
                {
                    switch (n++)
                    {
                        case 0: REQUIRE(group == "group1"); break;
                        case 1: REQUIRE(group == "group2"); break;
                        case 2: REQUIRE(group == "group3"); break;
                    }
                }
            }
        }

        WHEN("The file is closed")
        {
            file.close();

            THEN("Trying to open a group fails")
            {
                REQUIRE_THROWS(file.open_group("group1"));
            }
        }
    }
}


SCENARIO("Data types can be created", "[h5::Datatype]")
{
    REQUIRE(h5::detail::make_datatype_for(char()).size() == sizeof(char));
    REQUIRE(h5::detail::make_datatype_for(int()).size() == sizeof(int));
    REQUIRE(h5::detail::make_datatype_for(double()).size() == sizeof(double));
    REQUIRE(h5::detail::make_datatype_for(std::string("message")).size() == 7);
    REQUIRE(h5::detail::make_datatype_for(std::vector<int>()).size() == sizeof(int));
}


SCENARIO("Data spaces can be created", "[h5::Dataspace]")
{
    REQUIRE(h5::Dataspace().size() == 0);
    REQUIRE(h5::Dataspace::scalar().rank() == 0);
    REQUIRE(h5::Dataspace::scalar().size() == 1);
    REQUIRE(h5::Dataspace::scalar().select_all().size() == 1);
    REQUIRE(h5::Dataspace::scalar().select_none().size() == 1);
    REQUIRE(h5::Dataspace::scalar().select_all().selection_size() == 1);
    REQUIRE(h5::Dataspace::scalar().select_none().selection_size() == 0);
    REQUIRE(h5::Dataspace::simple(std::array<int, 3>{10, 10, 10}).rank() == 3);
    REQUIRE(h5::Dataspace::simple(std::array<int, 3>{10, 10, 10}).size() == 1000);
    REQUIRE(h5::Dataspace{10, 21}.size() == 210);
    REQUIRE(h5::Dataspace{10, 21}.selection_size() == 210);
    REQUIRE(h5::Dataspace{10, 21}.selection_lower() == std::vector<std::size_t>{0, 0});
    REQUIRE(h5::Dataspace{10, 21}.selection_upper() == std::vector<std::size_t>{9, 20});
    REQUIRE_NOTHROW(h5::Dataspace::scalar().select_all());

    GIVEN("A data space constructed from an nd::selector object")
    {
        auto sel = nd::selector<2>(100, 100);
        auto space = h5::Dataspace(sel);

        THEN("The data space and selector have the same size")
        {
            REQUIRE(space.size() == sel.size());
            REQUIRE(space.size() == sel.size());
            REQUIRE(space.extent() == std::vector<std::size_t>{100, 100});
            REQUIRE(space.selection_lower() == std::vector<std::size_t>{0, 0});
            REQUIRE(space.selection_upper() == std::vector<std::size_t>{99, 99});
            REQUIRE(space.selection_size() == space.size());
        }

        WHEN("We create another selector as a subset of the larger one")
        {
            auto _ = nd::axis::all();
            auto sub = sel.select(_|0|5, _|0|10);

            THEN("Another data space, with the same extent but smaller size")
            {
                auto sub_space = h5::Dataspace(sub);

                REQUIRE(sub_space.extent() == space.extent());
                REQUIRE(sub_space.selection_lower() == std::vector<std::size_t>{0, 0});
                REQUIRE(sub_space.selection_upper() == std::vector<std::size_t>{4, 9});
                REQUIRE(sub_space.selection_size() == sub.size());
            }
        }
    }
}


SCENARIO("Data sets can be created, read, and written to", "[h5::Dataset]")
{
    GIVEN("A file opened for writing, native double data type, and a scalar data space")
    {
        auto file = h5::File("test.h5", "w");
        auto type = h5::detail::make_datatype_for(double());
        auto space = h5::Dataspace::scalar();
        auto dset = file.require_dataset("data", type, space);

        THEN("The dataset exists in the file with expected properties")
        {
            REQUIRE(file.open_dataset("data").get_type() == type);
            REQUIRE(file.open_dataset("data").get_space().size() == space.size());
            REQUIRE_NOTHROW(file.open_dataset("data"));
            REQUIRE_NOTHROW(file.require_dataset("data", type, space));
            REQUIRE_THROWS(file.require_dataset("data", h5::detail::make_datatype_for(int()), space));
        }
    }

    GIVEN("A file opened for writing, native int data type, and a simple data space")
    {
        auto file = h5::File("test.h5", "w");
        auto type = h5::detail::make_datatype_for(int());
        auto space = h5::Dataspace::simple(std::array<int, 1>{4});
        auto dset = file.require_dataset("data", type, space);

        THEN("The dataset exists in the file with expected properties")
        {
            REQUIRE(file.open_dataset("data").get_type() == type);
            REQUIRE(file.open_dataset("data").get_space() == space);
            REQUIRE_NOTHROW(file.open_dataset("data"));
            REQUIRE_NOTHROW(file.require_dataset("data", type, space));
            REQUIRE_THROWS(file.require_dataset("data", h5::detail::make_datatype_for(double()), space));
        }

        WHEN("We have a std::vector<int>{1, 2, 3, 4}")
        {
            auto data = std::vector<int>{1, 2, 3, 4};

            THEN("It can be written to the data set and read back")
            {
                REQUIRE_NOTHROW(dset.write(data));
                REQUIRE(dset.read<std::vector<int>>() == data);
                REQUIRE_THROWS(dset.read<std::vector<double>>());
                REQUIRE_THROWS(dset.read<double>());
            }
        }

        WHEN("We have a std::vector<int>{1, 2, 3} or std::vector<double>{1, 2, 3, 4}")
        {
            auto data1 = std::vector<int>{1, 2, 3};
            auto data2 = std::vector<double>{1, 2, 3, 4};

            THEN("It cannot be written to the data set")
            {
                REQUIRE_THROWS(dset.write(data1));
                REQUIRE_THROWS(dset.write(data2));
            }
        }
    }

    GIVEN("A file opened for writing and a double")
    {
        auto data = 10.0;
        auto file = h5::File("test.h5", "w");
        auto type = h5::detail::make_datatype_for(data);
        auto space = h5::Dataspace::scalar();
        auto dset = file.require_dataset("data", type, space);

        THEN("The double can be written to a scalar dataset")
        {
            REQUIRE_NOTHROW(file.require_dataset("data", type, space));
            REQUIRE_NOTHROW(dset.write(data));
            REQUIRE(dset.read<double>() == 10.0);
            REQUIRE_THROWS(dset.read<int>() == 10);
        }
    }

    GIVEN("A file opened for writing")
    {
        auto data = std::string("The string value");
        auto file = h5::File("test.h5", "w");
        auto type = h5::detail::make_datatype_for(data);
        auto space = h5::Dataspace::scalar();
        auto dset = file.require_dataset("data1", type, space);

        THEN("A string can be written to a scalar dataset")
        {
            REQUIRE_NOTHROW(file.require_dataset("data1", type, space));
            REQUIRE_NOTHROW(dset.write(data));
            REQUIRE_THROWS(dset.read<int>());
            REQUIRE(dset.read<std::string>() == "The string value");
        }

        WHEN("A string, int, and double are written directly to the file")
        {
            file.write("data2", data);
            file.write("data3", 10.0);
            file.write("data4", 11);

            THEN("They can be read back out again")
            {
                REQUIRE(file.read<std::string>("data2") == data);
                REQUIRE(file.read<double>("data3") == 10.0);
                REQUIRE(file.read<int>("data4") == 11);
            }
        }
    }

    GIVEN("A file opened for writing")
    {
        auto file = h5::File("test.h5", "w");

        WHEN("An int and double vector are written to it")
        {
            auto data1 = std::vector<int>{1, 2, 3, 4};
            auto data2 = std::vector<double>{1, 2, 3};

            file.write("data1", data1);
            file.write("data2", data2);

            THEN("They can be read back again")
            {
                REQUIRE(file.read<decltype(data1)>("data1") == data1);
                REQUIRE(file.read<decltype(data2)>("data2") == data2);
            }

            THEN("Trying to read the wrong type throws")
            {
                REQUIRE_THROWS(file.read<decltype(data2)>("data1")); // mismatched data types
                REQUIRE_THROWS(file.read<decltype(data1)>("data2"));
            }
        }
    }
}


SCENARIO("Data sets can be selected on using ndarray syntax", "[h5::Dataspace] [nd::selector]")
{
    using D   = std::vector<double>;
    auto _    = nd::axis::all();
    auto file = h5::File("test.h5", "w");
    auto dset = file.require_dataset<double>("data", {5});

    dset.write(D{0, 1, 2, 3, 4});

    REQUIRE(dset.read<D>(nd::selector<1>(5).slice(0, 2, 1)) == D{0, 1});
    REQUIRE(dset.read<D>(nd::make_selector(_|0|2)) == D{0, 1});
    REQUIRE(dset.read<D>(nd::make_selector(_|0|4|2)) == D{0, 2});
    REQUIRE_THROWS(dset.read<D>(nd::make_selector(_|0|6)));   // out-of-bounds
    REQUIRE_THROWS(dset.read<D>(nd::make_selector(_|2|8|2))); // out-of-bounds
    REQUIRE_THROWS(dset.read<D>(nd::make_selector(_, _)));    // wrong rank
}


SCENARIO("nd::ndarray interaction works", "[h5::Dataset] [nd::ndarray]")
{
    auto _ = nd::axis::all();

    GIVEN("A file open for writing and a 1D array")
    {
        auto file = h5::File("test.h5", "w");
        auto data = nd::arange<double>(100);

        THEN("The array can be written to the file and read back again")
        {
            REQUIRE_NOTHROW(file.write("data", data));
            REQUIRE((file.read<decltype(data)>("data") == data).all());
        }

        WHEN("The array is resized to a 2D array")
        {
            auto data2 = data.reshape(10, 10);

            THEN("The array can still be written to the file and read back again")
            {
                REQUIRE_NOTHROW(file.write("data2", data2));
                REQUIRE((file.read<decltype(data2)>("data2") == data2).all());
            }

            THEN("A subset of the 2d array can be written and read back")
            {
                auto data3 = data2.select(0, _);
                REQUIRE_NOTHROW(file.write("data3", data3));
                REQUIRE((file.read<decltype(data3)>("data3") == data3).all());
            }
        }
    }
}

#endif // TEST_NDH5
