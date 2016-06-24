prog := reduce-kchain
prog_objs := reduce-kchain.o

CC := mpicc
CFLAGS := -std=c99 -Wall -O2
LDFLAGS := -lm

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

reduce-kchain.o: reduce-kchain.c

clean:
	@rm -rf *.o $(prog)
