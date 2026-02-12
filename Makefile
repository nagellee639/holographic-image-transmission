NVCC = nvcc
NVCC_FLAGS = -O3 -shared -Xcompiler -fPIC -std=c++11 -lcurand -lcublas -lm -DBUILD_LIB

default: libholo.so

libholo.so: holo_cuda.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f libholo.so
