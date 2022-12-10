#define gpuErrorCheck(ans)                    \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpu.cuh"
#include "defines.h"
#include "cuda_runtime_api.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__device__ bool isInRowGpu(int row, int number, int *board)
{
    for (int col = 0; col < BOARD_SIZE; col++)
    {
        if (board[row * BOARD_SIZE + col] == number)
            return true;
    }

    return false;
}

__device__ bool isInColumnGpu(int column, int number, int *boards)
{
    for (int row = 0; row < BOARD_SIZE; row++)
    {
        if (boards[row * BOARD_SIZE + column] == number)
            return true;
    }

    return false;
}

__device__ bool isInBoxGpu(int boxStartRow, int boxStartColumn, int number, int *board)
{
    for (int row = 0; row < BOX_SIZE; row++)
    {
        for (int col = 0; col < BOX_SIZE; col++)
        {
            if (board[boxStartRow * BOARD_SIZE + row * BOARD_SIZE + boxStartColumn + col] == number)
                return true;
        }
    }

    return false;
}

__device__ bool isValidPlace(int row, int column, int number, int *boards, int idx)
{
    int *board = boards + idx * (BOARD_SIZE * BOARD_SIZE);
    return !isInRowGpu(row, number, board) && !isInColumnGpu(column, number, board) && !isInBoxGpu(row - row % BOX_SIZE, column - column % BOX_SIZE, number, board);
}

__device__ void clearVisited(bool *visited)
{
    for (int i = 0; i < BOARD_SIZE; i++)
    {
        visited[i] = false;
    }
}

__device__ bool isRowValid(int row, const int *board)
{
    bool visited[BOARD_SIZE] = {false};

    for (int i = 0; i < BOARD_SIZE; i++)
    {
        int num = board[row * BOARD_SIZE + i];

        if (num != 0)
        {
            if (visited[num - 1])
            {
                return false;
            }
            else
            {
                visited[num - 1] = true;
            }
        }
    }

    return true;
}

__device__ bool isColumnValid(int column, const int *board)
{
    bool visited[BOARD_SIZE] = {false};

    for (int i = 0; i < BOARD_SIZE; i++)
    {
        int num = board[i * BOARD_SIZE + column];

        if (num != 0)
        {
            if (visited[num - 1])
            {
                return false;
            }
            else
            {
                visited[num - 1] = true;
            }
        }
    }

    return true;
}

__device__ bool isBoxValid(int rowIdx, int columnIdx, const int *board)
{
    bool visited[BOARD_SIZE] = {false};
    for (int i = 0; i < BOX_SIZE; i++)
    {
        for (int j = 0; j < BOX_SIZE; j++)
        {
            int num = board[(rowIdx * BOX_SIZE + i) * BOARD_SIZE + (columnIdx * BOX_SIZE + j)];

            if (num != 0)
            {
                if (visited[num - 1])
                {
                    return false;
                }
                else
                {
                    visited[num - 1] = true;
                }
            }
        }
    }

    return true;
}

__device__ bool isBoardValid(const int *board, int idx)
{
    int row = idx / BOARD_SIZE;
    int column = idx % BOARD_SIZE;

    int rowIdx = row / BOX_SIZE;
    int columnIdx = column / BOX_SIZE;

    if ((board[idx] < 1) || (board[idx] > 9))
    {
        return false;
    }

    return isRowValid(row, board) && isColumnValid(column, board) && isBoxValid(rowIdx, columnIdx, board);
}

// each thread generates new boards from array of current boards with valid inserts
__global__ void bfs(
    int *currentBoards,
    unsigned int currentBoardsCount,
    int *nextBoards,
    unsigned int *nextBoardIdx,
    int *emptySpaces,
    int *emptySpacesCounts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int startIdx = idx * BOARD_SIZE * BOARD_SIZE;
    int endIdx = (idx * BOARD_SIZE * BOARD_SIZE) + BOARD_SIZE * BOARD_SIZE;
    bool foundEmptyPlace = false;

    // finish if ran out of boards for current thread
    while (idx < currentBoardsCount)
    {
        foundEmptyPlace = false;

        for (int i = startIdx; i < endIdx && !foundEmptyPlace; i++)
        {
            if (currentBoards[i] == 0)
            {
                int shift = i - BOARD_SIZE * BOARD_SIZE * idx;

                // row and column for current empty place
                int currentRow = shift / BOARD_SIZE;
                int currentColumn = shift % BOARD_SIZE;
                foundEmptyPlace = true;

                for (int possibleNumber = 1; possibleNumber <= 9; possibleNumber++)
                {
                    if (isValidPlace(currentRow, currentColumn, possibleNumber, currentBoards, idx))
                    {
                        int nextIdx = atomicAdd(nextBoardIdx, 1);
                        int emptyCount = 0;
                        foundEmptyPlace = true;

                        for (int row = 0; row < BOARD_SIZE; row++)
                        {
                            for (int column = 0; column < BOARD_SIZE; column++)
                            {
                                int nextBoardIdx = nextIdx * (BOARD_SIZE * BOARD_SIZE) + row * BOARD_SIZE + column;
                                int currentBoardIdx = idx * (BOARD_SIZE * BOARD_SIZE) + row * BOARD_SIZE + column;

                                // copy currentBoard to nextBoard
                                nextBoards[nextBoardIdx] = currentBoards[currentBoardIdx];
                                if (currentBoards[currentBoardIdx] == 0 && !(row == currentRow && column == currentColumn))
                                {
                                    emptySpaces[emptyCount + BOARD_SIZE * BOARD_SIZE * nextIdx] = row * BOARD_SIZE + column;
                                    emptyCount++;
                                }
                            }
                        }
                        emptySpacesCounts[nextIdx] = emptyCount;
                        nextBoards[nextIdx * (BOARD_SIZE * BOARD_SIZE) + currentRow * BOARD_SIZE + currentColumn] = possibleNumber;
                    }
                }
            }
        }
        // if idx is less than currentBoardCount find next board in currentBoards
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void dfs(int *currentBoards,
                    unsigned int currentBoardsCount,
                    int *emptySpaces,
                    int *emptySpacesCounts,
                    int *finished,
                    int *result)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    int *currentBoard;
    int *currentEmptySpaces;
    int currentEmptySpacesCount;

    while ((*finished == 0) && (index < currentBoardsCount))
    {
        int emptyIndex = 0;

        currentBoard = currentBoards + index * BOARD_SIZE * BOARD_SIZE;
        currentEmptySpaces = emptySpaces + index * BOARD_SIZE * BOARD_SIZE;
        currentEmptySpacesCount = emptySpacesCounts[index];

        // check specific board
        while ((emptyIndex >= 0) && (emptyIndex < currentEmptySpacesCount))
        {
            currentBoard[currentEmptySpaces[emptyIndex]]++;

            if (isBoardValid(currentBoard, currentEmptySpaces[emptyIndex]))
            {
                emptyIndex++;
            }
            // if board is invalid and we checked all possibilities mark current place in board as empty and go back in empty indexes
            else if (currentBoard[currentEmptySpaces[emptyIndex]] >= 9)
            {
                currentBoard[currentEmptySpaces[emptyIndex]] = 0;
                emptyIndex--;
            }
        }

        // if all empty spaces have been filled solution was found
        if (emptyIndex == currentEmptySpacesCount)
        {
            *finished = 1;

            for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++)
            {
                result[i] = currentBoard[i];
            }
        }

        // go to next board
        index += gridDim.x * blockDim.x;
    }
}

void swapBoards(int *&currentBoards, int *&nextBoards)
{
    int *temp = currentBoards;
    currentBoards = nextBoards;
    nextBoards = temp;
}

float solveWithGpu(int *sudoku, int *result)
{
    clock_t executionStart, executionEnd, copyingStart, copyingEnd;
    // boards from which next set of boards will be generated
    int *currentBoards;
    // set of boards generated from currentBoards
    int *nextBoards;
    // index of next free space for generated board in nextBoards
    unsigned int *boardIdx;
    // flag if solution was found
    int *finished;
    // solved sudoku
    int *gpuResult;
    // number of boards in currentBoards
    unsigned int boardCount = 1;
    // indexes of empty spaces in specific board
    int *emptySpaces;
    // number of empty spaces in specific board
    int *emptySpacesCounts;
    // max size for all boards
    const double maxBoardsSize = BOARD_SIZE * BOARD_SIZE * MAX_BOARDS;
    float executionTime;
    float dataCopyingTime;

    std::copy(sudoku, sudoku + BOARD_SIZE * BOARD_SIZE, result);

    gpuErrorCheck(cudaFree(0));
    gpuErrorCheck(cudaSetDevice(0));

    gpuErrorCheck(cudaMalloc(&nextBoards, maxBoardsSize * sizeof(int)));
    gpuErrorCheck(cudaMalloc(&currentBoards, maxBoardsSize * sizeof(int)));
    gpuErrorCheck(cudaMalloc(&boardIdx, sizeof(unsigned int)));
    gpuErrorCheck(cudaMalloc(&finished, sizeof(int)));
    gpuErrorCheck(cudaMalloc(&gpuResult, BOARD_SIZE * BOARD_SIZE * sizeof(int)));
    gpuErrorCheck(cudaMalloc(&emptySpaces, maxBoardsSize * sizeof(int)));
    gpuErrorCheck(cudaMalloc(&emptySpacesCounts, (maxBoardsSize / (BOARD_SIZE * BOARD_SIZE) + 1) * sizeof(int)));

    gpuErrorCheck(cudaMemset(boardIdx, 0, sizeof(int)));
    gpuErrorCheck(cudaMemset(finished, 0, sizeof(int)));
    gpuErrorCheck(cudaMemset(nextBoards, 0, maxBoardsSize * sizeof(int)));
    gpuErrorCheck(cudaMemset(currentBoards, 0, maxBoardsSize * sizeof(int)));
    gpuErrorCheck(cudaMemset(gpuResult, 0, BOARD_SIZE * BOARD_SIZE * sizeof(int)));

    copyingStart = clock();
    gpuErrorCheck(cudaMemcpy(currentBoards, result, BOARD_SIZE * BOARD_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    copyingEnd = clock();

    dataCopyingTime = ((float)(copyingEnd - copyingStart)) / (CLOCKS_PER_SEC / 1000);

    executionStart = clock();

    for (int i = 0; i < ITERATIONS_COUNT; i++)
    {
        // start adding boards from 0 index in nextBoards
        gpuErrorCheck(cudaMemset(boardIdx, 0, sizeof(unsigned int)));

        // find next array of possible boards from currentBoards, and add them to nextBoards
        bfs<<<BLOCKS, THREADS>>>(
            currentBoards,
            boardCount,
            nextBoards,
            boardIdx,
            emptySpaces,
            emptySpacesCounts);

        // use nextBoards as currentBoards in next iteration
        swapBoards(currentBoards, nextBoards);

        copyingStart = clock();
        // get number of possible generated boards
        gpuErrorCheck(cudaMemcpy(&boardCount, boardIdx, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        copyingEnd = clock();
        dataCopyingTime += ((float)(copyingEnd - copyingStart)) / (CLOCKS_PER_SEC / 1000);
    }
    dfs<<<BLOCKS, THREADS>>>(currentBoards, boardCount, emptySpaces, emptySpacesCounts, finished, gpuResult);

    executionEnd = clock();

    executionTime = ((float)(executionEnd - executionStart)) / (CLOCKS_PER_SEC / 1000);

    copyingStart = clock();
    gpuErrorCheck(cudaMemcpy(result, gpuResult, BOARD_SIZE * BOARD_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
    copyingEnd = clock();
    dataCopyingTime += ((float)(copyingEnd - copyingStart)) / (CLOCKS_PER_SEC / 1000);

    std::cout << "Copying data took: " << dataCopyingTime << " ms\n";

    gpuErrorCheck(cudaFree(nextBoards));
    gpuErrorCheck(cudaFree(currentBoards));
    gpuErrorCheck(cudaFree(boardIdx));
    gpuErrorCheck(cudaFree(finished));
    gpuErrorCheck(cudaFree(gpuResult));
    gpuErrorCheck(cudaFree(emptySpaces));
    gpuErrorCheck(cudaFree(emptySpacesCounts));

    return executionTime;
}
