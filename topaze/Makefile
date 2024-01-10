
.PHONY: clean status

all: dp lightning

status:
	ccc_mpp -u lecomtej

dp:
	ccc_msub ./ccc_dp.sh

lightning:
	ccc_msub ./ccc_lightning.sh

clean:
	rm -rf *.o *.e mlruns
