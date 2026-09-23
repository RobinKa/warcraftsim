# Cross-compile the in-game helper (32-bit, matches Warcraft III 1.29) with mingw-w64.
CC := i686-w64-mingw32-gcc
OUT ?= ../build/shim
CFLAGS := -O2 -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type -std=gnu11 -municode
LDFLAGS := -static-libgcc -s

all: $(OUT)/w3shim.dll $(OUT)/w3launch.exe

MINHOOK := ../third_party/minhook
MINHOOK_SRC := $(MINHOOK)/src/buffer.c $(MINHOOK)/src/hook.c $(MINHOOK)/src/trampoline.c $(MINHOOK)/src/hde/hde32.c
SHIM_SRC := main.c clock.c iat.c sync.c turbo.c

$(OUT)/w3shim.dll: $(SHIM_SRC) w3shim.h | $(OUT)
	$(CC) $(filter-out -municode,$(CFLAGS)) -I$(MINHOOK)/include -shared -o $@ $(SHIM_SRC) \
		$(filter-out -Wextra,$(MINHOOK_SRC)) $(LDFLAGS) -lws2_32

$(OUT)/w3launch.exe: launcher.c | $(OUT)
	$(CC) $(CFLAGS) -o $@ launcher.c $(LDFLAGS)

$(OUT):
	mkdir -p $@

clean:
	rm -rf $(OUT)

.PHONY: all clean
