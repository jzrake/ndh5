CXXFLAGS = -std=c++14

HEADERS = ndh5.hpp

default: test main

main.o: ndh5.hpp

test.o: $(HEADERS)

test: test.o catch.o
	$(CXX) -o $@ $(CXXFLAGS) $^ -lhdf5

main: main.o
	$(CXX) -o $@ $(CXXFLAGS) $^ -lhdf5

clean:
	$(RM) *.o test main
