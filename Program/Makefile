#Variables

NVCC = nvcc
NVCCFLAGS = -gencode arch=compute_86,code=sm_86 -g

NVCCLINKS = -lcurand

#Targets

all: bin/csg-perform bin/csg-analyze

bin/csg-perform: bin/util.o bin/eamsim.o bin/csg-perform.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(NVCCLINKS)

bin/csg-analyze: bin/util.o bin/eamana.o bin/csg-analyze.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

bin/csg-perform.o: src/csg-perform.cu bin/eamsim.o
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

bin/csg-analyze.o: src/csg-analyze.cu bin/eamana.o
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

bin/eamsim.o: src/eamsim.cu src/eamsim.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

bin/eamana.o: src/eamana.cu src/eamana.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

bin/util.o: src/util.cu src/util.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -r bin/*
