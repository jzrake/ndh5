#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <hdf5.h>




// ============================================================================
namespace h5 // H5_API_START
{
    class Link;
    class File;
    class Group;
    class Dataset;
    class Datatype;
    class Dataspace;

    enum class Intent { rdwr, rdonly, swmr_write, swmr_read };
    enum class Object { file, group, dataset };

    template<typename T> static inline Datatype make_datatype(std::size_t count=1);
    template<typename T> static inline Datatype make_datatype_for(const T& val);

    namespace detail {
        static inline herr_t get_last_error(unsigned, const H5E_error2_t*, void*);
        static inline hid_t check(hid_t result);
        template<typename T> static inline const void* scalar_address(const T& val);
    }
} // H5_API_END




// ============================================================================
herr_t h5::detail::get_last_error(unsigned n, const H5E_error2_t *err, void *data)
{
    if (n == 0)
    {
        *static_cast<H5E_error2_t*>(data) = *err;
    }
    return 0;
}

hid_t h5::detail::check(hid_t result)
{
    if (result < 0)
    {
        H5E_error2_t err;
        hid_t eid = H5Eget_current_stack();
        H5Ewalk(eid, H5E_WALK_UPWARD, get_last_error, &err);
        H5Eclear(eid);
        H5Eclose_stack(eid);
        throw std::invalid_argument(err.desc);
    }
    return result;
}

template<> const void* h5::detail::scalar_address<std::string>(const std::string& val)
{
    return val.data();
}

template<typename T> const void* h5::detail::scalar_address(const T& val)
{
    return &val;
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
    template<typename>
    friend Datatype make_datatype(std::size_t);

    template<typename T>
    friend Datatype make_datatype_for(const T&);

    friend class Link;
    friend class Dataset;

    Datatype(hid_t id) : id(id) {}
    hid_t id = -1;
};




// ============================================================================
template<> h5::Datatype h5::make_datatype<char>(std::size_t count)
{
    return Datatype(H5Tcopy(H5T_C_S1)).with_size(count * sizeof(char));
}

template<> h5::Datatype h5::make_datatype<int>(std::size_t count)
{
    return Datatype(H5Tcopy(H5T_NATIVE_INT)).with_size(count * sizeof(int));
}

template<> h5::Datatype h5::make_datatype<double>(std::size_t count)
{
    return Datatype(H5Tcopy(H5T_NATIVE_DOUBLE)).with_size(count * sizeof(double));
}

template<> h5::Datatype h5::make_datatype_for<std::string>(const std::string& val)
{
    return make_datatype<char>(val.size());
}

template<typename T> h5::Datatype h5::make_datatype_for(const T& val)
{
    return make_datatype<T>();
}




// ============================================================================
class h5::Dataspace
{
public:
    static Dataspace scalar()
    {
        return H5Screate(H5S_SCALAR);
    }

    template<typename Container>
    static Dataspace simple(Container dims)
    {
        auto hdims = std::vector<hsize_t>(dims.begin(), dims.end());
        return H5Screate_simple(hdims.size(), &hdims[0], nullptr);
    }

    Dataspace() {}

    Dataspace(const Dataspace& other)
    {
        id = H5Scopy(other.id);
    }

    ~Dataspace()
    {
        close();
    }

    Dataspace& operator=(const Dataspace& other)
    {
        close();
        id = H5Scopy(other.id);
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
            H5Sclose(id);
        }
    }

    std::size_t rank() const
    {
        return id == -1 ? 0 : H5Sget_simple_extent_ndims(id);        
    }

    std::size_t size() const
    {
        return id == -1 ? 0 : H5Sget_simple_extent_npoints(id);
    }

    std::size_t selection_size() const
    {
        return id == -1 ? 0 : H5Sget_select_npoints(id);
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

    Dataspace& select_hyperslab(std::vector<std::size_t> start,
                                std::vector<std::size_t> count,
                                std::vector<std::size_t> skips,
                                std::vector<std::size_t> block)
    {
        if (start.size() != rank() ||
            count.size() != rank() ||
            skips.size() != rank() ||
            block.size() != rank())
        {
            throw std::invalid_argument("inconsistent selection sizes");
        }

        auto s = std::vector<hsize_t>(start.begin(), start.end());
        auto c = std::vector<hsize_t>(count.begin(), count.end());
        auto k = std::vector<hsize_t>(skips.begin(), skips.end());
        auto b = std::vector<hsize_t>(block.begin(), block.end());

        detail::check(H5Sselect_hyperslab(id, H5S_SELECT_SET, s.data(), c.data(), k.data(), b.data()));
        return *this;
    }

private:
    friend class Link;
    friend class Dataset;
    Dataspace(hid_t id) : id(id) {}
    hid_t id = -1;
};




// ============================================================================
class h5::Link
{
private:

    // ========================================================================
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

    void close(Object location)
    {
        if (id != -1)
        {
            switch (location)
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

    bool contains(const std::string& name, Object location) const
    {
        if (H5Lexists(id, name.data(), H5P_DEFAULT))
        {
            H5O_info_t info;
            H5Oget_info_by_name(id, name.data(), &info, H5P_DEFAULT);

            switch (location)
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

    Link create_dataset(const std::string& name, const Datatype& type, const Dataspace& space)
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
                throw std::overflow_error("location names longer than 1024 are not supported");
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
    friend class File;
    friend class Group;
    friend class Dataset;

    hid_t id = -1;
};




// ============================================================================
class h5::Dataset final
{
public:
    Dataset()
    {
    }

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
    void write_scalar(const T& value)
    {
        auto data = detail::scalar_address(value);
        auto type = check_compatible(make_datatype_for(value));
        auto mspace = Dataspace::scalar();
        auto fspace = get_space();
        detail::check(H5Dwrite(link.id, type.id, mspace.id, fspace.id, H5P_DEFAULT, data));
    }

    template<typename T>
    void write(const T& data)
    {
        auto type = check_compatible(make_datatype<typename T::value_type>());
        auto mspace = Dataspace::simple(std::vector<std::size_t>{data.size()});
        auto fspace = get_space();
        detail::check(H5Dwrite(link.id, type.id, mspace.id, fspace.id, H5P_DEFAULT, &data[0]));
    }

    template<typename T>
    T read_scalar() const
    {
        auto data = T();
        auto type = check_compatible(make_datatype<T>());
        auto mspace = Dataspace::scalar();
        auto fspace = get_space();
        detail::check(H5Dread(link.id, type.id, mspace.id, fspace.id, H5P_DEFAULT, &data));
        return data;
    }

    template<typename T>
    T read()
    {
        auto data = T(get_space().size());
        auto type = check_compatible(make_datatype<typename T::value_type>());
        auto mspace = Dataspace::simple(std::vector<std::size_t>{data.size()});
        auto fspace = get_space();
        detail::check(H5Dread(link.id, type.id, mspace.id, fspace.id, H5P_DEFAULT, &data[0]));
        return data;
    }

private:
    Datatype check_compatible(const Datatype& type) const
    {
        if (type != get_type())
        {
            throw std::invalid_argument("source and target have different data types");
        }
        return type;
    }

    friend class File;
    friend class Group;
    friend class Location;
    Dataset(Link link) : link(std::move(link)) {}
    Link link;
};

template<>
std::string h5::Dataset::read_scalar<std::string>() const
{
    auto data = std::string(get_type().size(), '\0');
    auto type = check_compatible(make_datatype_for(data));
    auto mspace = Dataspace::scalar();
    auto fspace = get_space();
    detail::check(H5Dread(link.id, type.id, mspace.id, fspace.id, H5P_DEFAULT, &data[0]));
    return data;
}




// ============================================================================
class h5::Group final
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

    Group operator[](const std::string& name)
    {
        return require_group(name);
    }

    Group open_group(const std::string& name)
    {
        return link.open_group(name);
    }

    Dataset open_dataset(const std::string& name)
    {
        return link.open_dataset(name);
    }

    Group require_group(const std::string& name)
    {
        return link.contains(name, Object::group) ? open_group(name) : link.create_group(name);
    }

    Dataset require_dataset(const std::string& name, const Datatype& type, const Dataspace& space)
    {
        if (link.contains(name, Object::dataset))
        {
            auto dset = open_dataset(name);

            if (dset.get_type() == type && dset.get_space() == space)
            {
                return dset;
            }
            throw std::invalid_argument("data set with different type or space already exists");
        }
        return link.create_dataset(name, type, space);
    }

private:
    friend class File;
    Group(Link link) : link(std::move(link)) {}
    Link link;
};




// ============================================================================
class h5::File final
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

    File& operator=(File&& other)
    {
        link = std::move(other.link);
        return *this;
    }

    ~File()
    {
        close();
    }

    Intent intent() const
    {
        unsigned intent;
        detail::check(H5Fget_intent(link.id, &intent));

        if (intent == H5F_ACC_RDWR)       return Intent::rdwr;
        if (intent == H5F_ACC_RDONLY)     return Intent::rdonly;
        if (intent == H5F_ACC_SWMR_WRITE) return Intent::swmr_write;
        if (intent == H5F_ACC_SWMR_READ)  return Intent::swmr_read;

        throw;
    }

    void close()
    {
        link.close(Object::file);
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

    Group operator[](const std::string& name)
    {
        return require_group(name);
    }

    Group open_group(const std::string& name)
    {
        return link.open_group(name);
    }

    Dataset open_dataset(const std::string& name)
    {
        return link.open_dataset(name);
    }

    Group require_group(const std::string& name)
    {
        return link.contains(name, Object::group) ? open_group(name) : link.create_group(name);
    }

    Dataset require_dataset(const std::string& name, const Datatype& type, const Dataspace& space)
    {
        if (link.contains(name, Object::dataset))
        {
            auto dset = open_dataset(name);

            if (dset.get_type() == type && dset.get_space() == space)
            {
                return dset;
            }
            throw std::invalid_argument("data set with different type or space already exists");
        }
        return link.create_dataset(name, type, space);
    }

private:
    File(Link link) : link(std::move(link)) {}
    Link link;
};




// ============================================================================
#ifdef TEST_NDH5
#include "catch.hpp"
#include <array>




SCENARIO("files can be created", "[h5::File]")
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


SCENARIO("groups can be created in files", "[h5::Group]")
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
    REQUIRE(h5::make_datatype<char>(100).size() == 100);
    REQUIRE(h5::make_datatype<int>().size() == sizeof(int));
    REQUIRE(h5::make_datatype<double>().size() == sizeof(double));
    REQUIRE(h5::make_datatype_for(std::string("message")).size() == 7);
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
    REQUIRE(h5::Dataspace::simple(std::vector<size_t>{10, 21}).size() == 210);
    REQUIRE_THROWS(h5::Dataspace().select_all());
    REQUIRE_NOTHROW(h5::Dataspace::scalar().select_all());
}


SCENARIO("Data sets can be created nd written to", "[h5::Dataset]")
{
    GIVEN("A file opened for writing, native double data type, and a scalar data space")
    {
        auto file = h5::File("test.h5", "w");
        auto type = h5::make_datatype<double>();
        auto space = h5::Dataspace::scalar();
        auto dset = file.require_dataset("data", type, space);

        THEN("The dataset exists in the file with expected properties")
        {
            REQUIRE(file.open_dataset("data").get_type() == type);
            REQUIRE(file.open_dataset("data").get_space().size() == space.size());
            REQUIRE_NOTHROW(file.open_dataset("data"));
            REQUIRE_NOTHROW(file.require_dataset("data", type, space));
            REQUIRE_THROWS(file.require_dataset("data", h5::make_datatype<int>(), space));
        }
    }

    GIVEN("A file opened for writing, native int data type, and a simple data space")
    {
        auto file = h5::File("test.h5", "w");
        auto type = h5::make_datatype<int>();
        auto space = h5::Dataspace::simple(std::array<int, 1>{4});
        auto dset = file.require_dataset("data", type, space);

        THEN("The dataset exists in the file with expected properties")
        {
            REQUIRE(file.open_dataset("data").get_type() == type);
            REQUIRE(file.open_dataset("data").get_space() == space);
            REQUIRE_NOTHROW(file.open_dataset("data"));
            REQUIRE_NOTHROW(file.require_dataset("data", type, space));
            REQUIRE_THROWS(file.require_dataset("data", h5::make_datatype<double>(), space));
        }

        WHEN("We have a std::vector<int>{1, 2, 3, 4}")
        {
            auto data = std::vector<int>{1, 2, 3, 4};

            THEN("It can be written to the data set and read back")
            {
                REQUIRE_NOTHROW(dset.write(data));
                REQUIRE(dset.read<std::vector<int>>() == data);
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
        auto type = h5::make_datatype_for(data);
        auto space = h5::Dataspace::scalar();
        auto dset = file.require_dataset("data", type, space);

        THEN("The double can be written to a scalar dataset")
        {
            REQUIRE_NOTHROW(file.require_dataset("data", type, space));
            REQUIRE_NOTHROW(dset.write_scalar(data));
            REQUIRE(dset.read_scalar<double>() == 10.0);
            REQUIRE_THROWS(dset.read_scalar<int>() == 10);
        }
    }

    GIVEN("A file opened for writing and a string")
    {
        auto data = std::string("The string value");
        auto file = h5::File("test.h5", "w");
        auto type = h5::make_datatype_for(data);
        auto space = h5::Dataspace::scalar();
        auto dset = file.require_dataset("data", type, space);

        THEN("The string can be written to a scalar dataset")
        {
            REQUIRE_NOTHROW(file.require_dataset("data", type, space));
            REQUIRE_NOTHROW(dset.write_scalar(data));
            REQUIRE_THROWS(dset.read_scalar<int>());
            REQUIRE(dset.read_scalar<std::string>() == "The string value");
        }
    }
}

#endif // TEST_NDH5
