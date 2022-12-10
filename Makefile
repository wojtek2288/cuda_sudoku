all: sudoku

sudoku: 
	nvcc -o sudoku main.cu gpu.cu cpu.cpp io.cpp

.PHONY: clean all

clean:
	rm sudoku
