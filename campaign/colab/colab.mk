GPU ?= T4
MAX_CU ?=
COLAB := python3 -I $(CURDIR)/colab/colab.py

.PHONY: colab-probe colab-list colab-smoke colab-bench colab-cleanup colab-test

colab-probe:
	@$(COLAB) probe

colab-list:
	@$(COLAB) list

colab-smoke:
	@test -n "$(MAX_CU)" || { echo "colab-smoke needs MAX_CU, e.g. MAX_CU=0.25" >&2; exit 2; }
	@$(COLAB) smoke --gpu $(GPU) --max-cu $(MAX_CU)

colab-bench:
	@test -n "$(MAX_CU)" || { echo "colab-bench needs MAX_CU, e.g. MAX_CU=0.5" >&2; exit 2; }
	@test -n "$(NAMES)" || { echo "colab-bench needs NAMES; see make -C campaign colab-list" >&2; exit 2; }
	@$(COLAB) bench --gpu $(GPU) --max-cu $(MAX_CU) --names '$(NAMES)'

colab-cleanup:
	@$(COLAB) cleanup

colab-test:
	python3 -I -m unittest discover -s $(CURDIR)/tests -p 'test_colab.py'
