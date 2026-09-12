# Lorenz-96 EnKF/LETKF twin experiment
#
# Modules:
#   rng.c     - random number generation
#   l96.c     - Lorenz-96 forward model (RK4)
#   linalg.c  - matrix inverse, Jacobi eigensolver, Gaspari-Cohn taper
#   filter.c  - EnKF (stochastic, localized gain) and LETKF analyses
#   diag.c    - diagnostics and CSV output helpers
#   main.c    - experiment driver

CC     = gcc
CFLAGS = -O2 -Wall -Wextra
LDLIBS = -lm

OBJS = main.o rng.o l96.o linalg.o filter.o diag.o

.PHONY: all clean

all: lorenz96_enkf

lorenz96_enkf: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f lorenz96_enkf $(OBJS)