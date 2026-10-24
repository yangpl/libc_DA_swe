CC = gcc
PYTHON ?= python3
CFLAGS = -O3 -std=c11 -Wall -Wextra -Wpedantic -Werror
LDLIBS = -lm

.PHONY: all test sanitize run clean
all: libswe.so

libswe.so: swe2d_4dvar.c swe2d_4dvar.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared $< $(LDFLAGS) $(LDLIBS) -o $@

build/test_kernel: tests/test_kernel.c swe2d_4dvar.c swe2d_4dvar.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

build/block_reference.o: tests/reference/block_swe.c swe2d_4dvar.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Dswe_forward=block_swe_forward -Dswe_evaluate=block_swe_evaluate -c $< -o $@

build/libblock_reference.so: tests/reference/block_swe.c swe2d_4dvar.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared $< $(LDFLAGS) $(LDLIBS) -o $@

build/test_checkpointing: tests/test_checkpointing.c swe2d_4dvar.c swe2d_4dvar.h build/block_reference.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $< build/block_reference.o $(LDFLAGS) $(LDLIBS) -o $@

test: libswe.so build/test_kernel build/test_checkpointing
	$(PYTHON) -m unittest discover -s tests -v
	./build/test_kernel
	./build/test_checkpointing

sanitize:
	@mkdir -p build
	$(CC) $(CPPFLAGS) -O1 -g -std=c11 -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer tests/test_kernel.c $(LDLIBS) -o build/test_kernel_sanitize
	ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 ./build/test_kernel_sanitize
	$(CC) -O1 -g -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer -Dswe_forward=block_swe_forward -Dswe_evaluate=block_swe_evaluate -c tests/reference/block_swe.c -o build/block_reference_sanitize.o
	$(CC) -O1 -g -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer tests/test_checkpointing.c build/block_reference_sanitize.o $(LDLIBS) -o build/test_checkpointing_sanitize
	ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 ./build/test_checkpointing_sanitize

run:
	bash run.sh

clean:
	rm -f libswe.so build/test_kernel build/test_kernel_sanitize build/test_checkpointing build/test_checkpointing_sanitize build/block_reference.o build/block_reference_sanitize.o
	rm -f build/libblock_reference.so
