#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <iostream>

#include "io.h"
#include "defines.h"
#include "cpu.h"
#include "gpu.cuh"

void printSudoku(int *board)
{
    for (int i = 0; i < BOARD_SIZE; i++)
    {
        if (i % BOX_SIZE == 0)
        {
            std::cout << "-------------------------\n";
        }

        for (int j = 0; j < BOARD_SIZE; j++)
        {
            if (j % BOX_SIZE == 0)
            {
                std::cout << "| ";
            }

            std::cout << board[i * BOARD_SIZE + j] << " ";
        }

        std::cout << "|\n";
    }
    std::cout << "-------------------------\n";
}

void checkCpu(const int *sudoku, int *result)
{
    float timeTaken = solveWithCpu(sudoku, result);

    std::cout << "CPU solution: \n";
    printSudoku(result);
    std::cout << "Time taken for the cpu is: " << timeTaken << " ms\n";
}

void checkGpu(int *sudoku, int *result)
{
    float timeTaken = solveWithGpu(sudoku, result);

    std::cout << "GPU solution: \n";
    printSudoku(result);
    std::cout << "Time taken for the gpu is: " << timeTaken << " ms\n";
}

int main(int argc, char *argv[])
{
    int sudoku[BOARD_SIZE * BOARD_SIZE];
    int cpuResult[BOARD_SIZE * BOARD_SIZE];
    int gpuResult[BOARD_SIZE * BOARD_SIZE];

    FILE *inputFile;

    if (argc != 2)
    {
        std::cout << "Specify file with sudoku board";
        return EXIT_FAILURE;
    }

    if ((inputFile = fopen(argv[1], "r")) == NULL)
    {
        std::cout << "Could not open file: " << argv[1];
        return EXIT_FAILURE;
    }

    readSudokuFromFile(inputFile, sudoku);

    checkCpu(sudoku, cpuResult);
    std::cout << '\n';
    checkGpu(sudoku, gpuResult);
}