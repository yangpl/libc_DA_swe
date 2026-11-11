CC = gcc
PYTHON ?= python3
CFLAGS = -O3 -std=c11 -Wall -Wextra -Wpedantic -Werror
SOURCES = swe_model.c swe_primal.c
HEADERS = swe_model.h swe_primal.h

.PHONY: all test sanitize compare run clean
all: libswe.so

libswe.so: $(SOURCES) $(HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared $(SOURCES) $(LDFLAGS) -lm -o $@

build/test_forward: tests/test_forward.c $(SOURCES) $(HEADERS)
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. $< $(SOURCES) $(LDFLAGS) -lm -o $@

test: libswe.so build/test_forward
	$(PYTHON) -m unittest discover -s tests -v
	./build/test_forward

compare: libswe.so
	$(PYTHON) tests/compare_forward.py

sanitize:
	@mkdir -p build
	$(CC) -O1 -g -std=c11 -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer -I. tests/test_forward.c $(SOURCES) -lm -o build/test_forward_sanitize
	ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 ./build/test_forward_sanitize

run:
	bash run.sh

clean:
	rm -f libswe.so build/test_forward build/test_forward_sanitize
