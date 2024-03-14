#Variables

NVCC = nvcc

NVCCFLAGS = -gencode arch=compute_86,code=sm_86

NVCCLINKS = -lcurand

#Targets

bin/csg: bin/util.o bin/eamsim.o bin/main.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(NVCCLINKS)

bin/main.o: src/main.cu bin/eamsim.o
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

bin/eamsim.o: src/eamsim.cu src/eamsim.cuh bin/util.o
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

bin/util.o: src/util.cu src/util.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -r bin/*
