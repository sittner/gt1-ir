# SPDX-License-Identifier: GPL-2.0-only
CFLAGS ?= -O3 -ffast-math -mcpu=native -Wall
LDLIBS = -lasound -lfftw3f -lm -lpthread

all: gt1conv latency

gt1conv: gt1conv.c
latency: latency.c
	$(CC) -O2 -Wall -o $@ $< -lasound -lm

clean:
	rm -f gt1conv latency
