
# if using PNG files:
#PNGCONF=/usr/bin/libpng-config
ifdef PNGCONF
CFLAGS=-DPNG $(shell $(PNGCONF) --cflags)
LDFLAGS=$(shell $(PNGCONF) --ldflags)
OBJ=pngwriter.c
else

endif

all:    main cuda cuda-shared cuda-1D

main:	src/main.c $(OBJ)
	cc -o main $(CFLAGS) $< $(OBJ) $(LDFLAGS)

cuda: src/cuda.cu $(OBJ)
	nvcc -o $@ $<

cuda-shared: src/cuda-shared.cu $(OBJ)
	nvcc -o $@ $<

cuda-1D: src/cuda-1D.cu $(OBJ)
	nvcc -o $@ $<

clean:
	rm -f main *.o
