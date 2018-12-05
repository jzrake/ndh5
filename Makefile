-include Makefile.in

CXXFLAGS = -std=c++14 -Wextra -Wno-missing-braces $(INCLUDE)
HEADERS = ndh5.hpp

default: test main

main.o: ndh5.hpp

test.o: $(HEADERS)

test: test.o catch.o
	$(CXX) -o $@ $(CXXFLAGS) $^ -lhdf5 $(LIBRARY)

main: main.o
	$(CXX) -o $@ $(CXXFLAGS) $^ -lhdf5 $(LIBRARY)

clean:
	$(RM) *.o test main
