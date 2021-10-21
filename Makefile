###############################################################################
# Makefile for assignment 2, Parallel and Distributed Computing 2020.
###############################################################################

CC = mpicc
CFLAGS = -std=gnu99 -g -O3
LIBS = -lm

BIN = quicksort

all: $(BIN)

quicksort: quicksort.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

github: github2.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)
	
clean:
	$(RM) $(BIN)