CC = cc
CFLAGS = -std=c99 -Wall -Wextra -Werror -pedantic

.PHONY: all clean

all: main1

main1: main1.o reverse.o
	$(CC) $(CFLAGS) $^ -o $@

main1.o: main1.c reverse.h
	$(CC) $(CFLAGS) -c main1.c

reverse.o: reverse.c reverse.h
	$(CC) $(CFLAGS) -c reverse.c

clean:
	rm -f main1 main1.o reverse.o
