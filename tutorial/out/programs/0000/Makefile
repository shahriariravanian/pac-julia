# u_ubar_to_z_g_g flow:2,4,5,1, compiled by PacRunner's compile.
#   make         runner.o and pacops.o
#   make clean
# CFLAGS may be changed; PACFLAGS may not: the values are bit for bit the Julia runner's only
# without contracted multiply-adds (-ffp-contract=off).
CC ?= cc
CFLAGS ?= -O2
PACFLAGS = -std=c99 -ffp-contract=off
all: runner.o pacops.o

runner.o: runner.c runner.h pacops.h
	$(CC) $(CFLAGS) $(PACFLAGS) -c -o $@ runner.c

pacops.o: pacops.c pacops.h
	$(CC) $(CFLAGS) $(PACFLAGS) -c -o $@ pacops.c

clean:
	rm -f runner.o pacops.o

.PHONY: all clean
