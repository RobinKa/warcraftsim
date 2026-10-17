# Cross-compile the in-game helper (32-bit, matches Warcraft III 1.29) with mingw-w64.
CC := i686-w64-mingw32-gcc
OUT ?= ../build/shim
# (SSE float math: units.c computes reals as the JASS VM does, in 32 bits)
CFLAGS := -O2 -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type -std=gnu11 -municode -msse2 -mfpmath=sse
LDFLAGS := -static-libgcc -s

all: $(OUT)/w3shim.dll $(OUT)/w3launch.exe

MINHOOK := ../third_party/minhook
MINHOOK_SRC := $(MINHOOK)/src/buffer.c $(MINHOOK)/src/hook.c $(MINHOOK)/src/trampoline.c $(MINHOOK)/src/hde/hde32.c
SHIM_SRC := main.c clock.c iat.c sync.c turbo.c registry.c obs.c audio.c units.c render.c syncstat.c cuts.c

$(OUT)/w3shim.dll: $(SHIM_SRC) w3shim.h | $(OUT)
	$(CC) $(filter-out -municode,$(CFLAGS)) -I$(MINHOOK)/include -shared -o $@ $(SHIM_SRC) \
		$(filter-out -Wextra,$(MINHOOK_SRC)) $(LDFLAGS) -lws2_32 -lwinmm

$(OUT)/w3launch.exe: launcher.c | $(OUT)
	$(CC) $(CFLAGS) -o $@ launcher.c $(LDFLAGS)

# the build for playing back people's replays (warcraftsim.fullgame.replays): replay.c, W3SIM_REPLAY
OUT_REPLAY ?= ../build/shim-replay
replay: $(OUT_REPLAY)/w3shim.dll $(OUT_REPLAY)/w3launch.exe

$(OUT_REPLAY)/w3shim.dll: $(SHIM_SRC) replay.c w3shim.h | $(OUT_REPLAY)
	$(CC) $(filter-out -municode,$(CFLAGS)) -DW3SIM_REPLAY -I$(MINHOOK)/include -shared -o $@ $(SHIM_SRC) replay.c \
		$(filter-out -Wextra,$(MINHOOK_SRC)) $(LDFLAGS) -lws2_32 -lwinmm

$(OUT_REPLAY)/w3launch.exe: launcher.c | $(OUT_REPLAY)
	$(CC) $(CFLAGS) -o $@ launcher.c $(LDFLAGS)

$(OUT) $(OUT_REPLAY):
	mkdir -p $@

clean:
	rm -rf $(OUT) $(OUT_REPLAY)

.PHONY: all clean replay
