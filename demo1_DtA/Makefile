CC = mpicc
CFLAGS = -O3 -g -Wall 

# FFTW_INC =/home/pyang/software_alias/external/gcc/include/fftw # /applis/site/stow/gcc_4.4.6/fftw_3.3.3/include
# FFTW_LIB =/home/pyang/software_alias/external/gcc/lib64  # /applis/site/stow/gcc_4.4.6/fftw_3.3.3/lib

# NETCDF_INC =/home/pyang/software_alias/external/gcc/include/netcdf # /applis/site/stow/gcc_4.4.6/netcdf_3.3.3/include
# NETCDF_LIB =/home/pyang/software_alias/external/gcc/lib64  # /applis/site/stow/gcc_4.4.6/netcdf_3.3.3/lib


BIN = .
# LIB = -L$(FFTW_LIB) -lfftw3  -L$(NETCDF_LIB) -lnetcdf -lm -lmpi -fopenmp
# INC = -I. -I$(FFTW_INC) -I$(NETCDF_INC)
LIB = -lm -lmpi
INC = -I.
MAIN_SRC = cstd.c main.c swe.c
MAIN_OBJ = $(MAIN_SRC:.c=.o)
GRADCHECK_SRC = gradcheck.c swe.c
GRADCHECK_OBJ = $(GRADCHECK_SRC:.c=.o)
ADJCHECK_SRC = adjcheck.c swe.c
ADJCHECK_OBJ = $(ADJCHECK_SRC:.c=.o)

all: clean main

main:	$(MAIN_OBJ)
	$(CC) $(CFLAGS) -o $(BIN)/main $(MAIN_OBJ) $(LIB)

gradcheck: $(GRADCHECK_OBJ)
	$(CC) $(CFLAGS) -o $(BIN)/gradcheck $(GRADCHECK_OBJ) $(LIB)

adjcheck: $(ADJCHECK_OBJ)
	$(CC) $(CFLAGS) -o $(BIN)/adjcheck $(ADJCHECK_OBJ) $(LIB)

%.o: %.c 
	$(CC) $(CFLAGS) -c $^ -o $@ $(INC) $(LIB)

clean:
	find . -name "*.o"   -exec rm {} \;
	find . -name "*.c%"  -exec rm {} \;
	find . -name "*.bck" -exec rm {} \;
	find . -name "*~"    -exec rm {} \;
	find . -name "\#*"   -exec rm {} \;
	rm -f $(MAIN_OBJ) $(GRADCHECK_OBJ) $(ADJCHECK_OBJ) main gradcheck adjcheck
