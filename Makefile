.PHONY: test lint

ifeq ($(shell uname -s),Darwin)
test:
	zig build test test-golden -Dcpu=apple_m1

lint:
	zig build dev-tools test-kernel-contract -Doptimize=Debug -Dcpu=apple_m1
else
test:
	zig build -Dfixed-cpu test-host --test-timeout 20m

lint:
	zig build -Dfixed-cpu test-kernel-contract
endif
