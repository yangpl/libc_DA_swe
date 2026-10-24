CC = gcc
CFLAGS = -O3 -std=c11 -Wall -Wextra -Wpedantic -Werror
TAPENADE_HOME ?= $(HOME)/Install/tapenade_3.16
TAPENADE ?= $(TAPENADE_HOME)/bin/tapenade
ADKIT ?= $(TAPENADE_HOME)/ADFirstAidKit
export TAPENADE_HOME ADKIT
RUNTIME_DIR = build/tapenade_runtime
CPPFLAGS = -D_POSIX_C_SOURCE=200809L -I$(RUNTIME_DIR) -Itapenade_out
SOURCES = swe2d_4dvar.c swe_primal.c tapenade_out/swe_adjoint_b.c
RUNTIME_SOURCES = $(RUNTIME_DIR)/adStack.c $(RUNTIME_DIR)/adBinomial.c
RUNTIME_FILES = $(RUNTIME_SOURCES) $(addprefix $(RUNTIME_DIR)/,adStack.h adBinomial.h adComplex.h LICENSE.html manifest.json)
HEADERS = swe2d_4dvar.h swe_primal.h tapenade_out/swe_primal_b.h $(RUNTIME_DIR)/adStack.h $(RUNTIME_DIR)/adBinomial.h $(RUNTIME_DIR)/adComplex.h
OBJECTS = $(patsubst %.c,build/obj/%.o,$(SOURCES)) $(RUNTIME_SOURCES:.c=.o)
# Generated code/runtime contains benign unused arguments and old-style C.
ADFLAGS = $(filter-out -Werror,$(CFLAGS)) -Wno-unused-parameter -Wno-unused-variable

.PHONY: all test sanitize run adjoint-generate generate-adjoint check-generated clean FORCE
all: libswe.so
libswe.so: $(OBJECTS) | check-generated
	$(CC) -shared $(OBJECTS) -lm -pthread -o $@
# Check content and installation path on each build; unchanged files keep mtimes.
# Grouped targets run preparation once even with make -j.
$(RUNTIME_FILES) &: FORCE prepare_tapenade_runtime.py tapenade_alignment.patch
	python3 prepare_tapenade_runtime.py --adkit "$(ADKIT)"
FORCE:
build/obj/tapenade_out/%.o: tapenade_out/%.c $(HEADERS)
	@mkdir -p $(@D)
	$(CC) $(ADFLAGS) $(CPPFLAGS) -fPIC -c $< -o $@
$(RUNTIME_DIR)/%.o: $(RUNTIME_DIR)/%.c $(HEADERS)
	$(CC) $(ADFLAGS) $(CPPFLAGS) -fPIC -c $< -o $@
build/obj/%.o: %.c $(HEADERS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(CPPFLAGS) -fPIC -c $< -o $@
adjoint-generate generate-adjoint:
	TAPENADE="$(TAPENADE)" python3 generate_adjoint.py
check-generated:
	python3 check_generated.py
test: libswe.so check-generated build/checkpoint_schedule
	python3 -m unittest discover -s tests -v
	$(CC) $(CFLAGS) tests/test_kernel.c -lm -o build/swe_kernel_test
	build/swe_kernel_test
	$(CC) $(CFLAGS) tests/test_tapenade.c -L. -lswe -Wl,-rpath,'$$ORIGIN/..' -lm -pthread -o build/swe_tapenade_test
	build/swe_tapenade_test
	build/checkpoint_schedule
build/checkpoint_schedule: tests/checkpoint_schedule.c $(RUNTIME_DIR)/adBinomial.c $(RUNTIME_DIR)/adBinomial.h
	@mkdir -p $(@D)
	$(CC) $(ADFLAGS) $(CPPFLAGS) $< $(RUNTIME_DIR)/adBinomial.c -o $@
sanitize: check-generated $(RUNTIME_FILES)
	$(CC) -O1 -g -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer $(CPPFLAGS) tests/test_tapenade.c $(SOURCES) $(RUNTIME_SOURCES) -lm -pthread -o build/swe_tapenade_sanitize
	ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 build/swe_tapenade_sanitize
run: libswe.so
	bash run.sh
clean:
	rm -f libswe.so build/checkpoint_schedule build/swe_kernel_test build/swe_tapenade_test build/swe_tapenade_sanitize
	rm -rf build/obj $(RUNTIME_DIR) build/tapenade-generation
