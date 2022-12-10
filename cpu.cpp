#include <stdlib.h>
#include <algorithm>
#include <time.h>

#include "cpu.h"
#include "defines.h"

bool isInColumn(int col, int num, int *sudoku)
{
    for (int row = 0; row < BOARD_SIZE; row++)
    {
        if (sudoku[row * BOARD_SIZE + col] == num)
            return true;
    }

    return false;
}

bool isInRow(int row, int num, int *sudoku)
{
    for (int col = 0; col < BOARD_SIZE; col++)
    {
        if (sudoku[row * BOARD_SIZE + col] == num)
            return true;
    }

    return false;
}

bool isInBox(int boxStartRow, int boxStartColumn, int num, int *sudoku)
{
    for (int row = 0; row < BOX_SIZE; row++)
    {
        for (int col = 0; col < BOX_SIZE; col++)
        {
            if (sudoku[boxStartRow * BOARD_SIZE + row * BOARD_SIZE + boxStartColumn + col] == num)
                return true;
        }
    }

    return false;
}

bool findEmptyPlace(int &row, int &col, int *sudoku)
{
    for (row = 0; row < BOARD_SIZE; row++)
    {
        for (col = 0; col < BOARD_SIZE; col++)
        {
            if (sudoku[row * BOARD_SIZE + col] == 0)
                return true;
        }
    }

    return false;
}

bool isValidPlace(int row, int col, int num, int *sudoku)
{
    return !isInRow(row, num, sudoku) && !isInColumn(col, num, sudoku) && !isInBox(row - row % BOX_SIZE, col - col % BOX_SIZE, num, sudoku);
}

bool backtrackWithCpu(int *sudoku)
{
    int row, col;

    // find empty place and store indexes in row and col
    if (!findEmptyPlace(row, col, sudoku))
    {
        return true;
    }

    for (int num = 1; num <= 9; num++)
    {
        if (isValidPlace(row, col, num, sudoku))
        {
            // make a guess if place is valid for num
            sudoku[row * BOARD_SIZE + col] = num;

            if (backtrackWithCpu(sudoku))
            {
                // found solution
                return true;
            }

            // backtrack if guess was wrong
            sudoku[row * BOARD_SIZE + col] = 0;
        }
    }
    // no solution possible
    return false;
}

float solveWithCpu(const int *sudoku, int *result)
{
    clock_t start, end;
    std::copy(sudoku, sudoku + BOARD_SIZE * BOARD_SIZE, result);

    start = clock();
    backtrackWithCpu(result);
    end = clock();

    return (float)(end - start) / (CLOCKS_PER_SEC / 1000);
}