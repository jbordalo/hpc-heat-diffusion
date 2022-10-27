
# if using PNG files:
#PNGCONF=/usr/bin/libpng-config
ifdef PNGCONF
CFLAGS=-DPNG $(shell $(PNGCONF) --cflags)
LDFLAGS=$(shell $(PNGCONF) --ldflags)
OBJ=pngwriter.c
else

endif

all:    main

main:	src/main.c $(OBJ)
	cc -o main $(CFLAGS) $< $(OBJ) $(LDFLAGS)

clean:
	rm -f main *.o
