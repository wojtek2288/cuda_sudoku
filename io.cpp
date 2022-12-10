#include <stdlib.h>
#include <stdio.h>

#include "io.h"
#include "defines.h"

void readSudokuFromFile(FILE* file, int* sudoku)
{
    char sudokuRow[BOARD_SIZE + 1];

    for (int i = 0; i < BOARD_SIZE; i++)
    {
        if (fscanf(file, "%s", sudokuRow) < 0)
            ERR("fscanf");

        for (int j = 0; j < BOARD_SIZE; j++)
        {
            sudoku[i * BOARD_SIZE + j] = sudokuRow[j] - 48;
        }
    }
}

void writeSudokuToFile(FILE* file, int* sudoku)
{
    for (int i = 0; i < BOARD_SIZE; i++)
    {
        for (int j = 0; j < BOARD_SIZE; j++)
        {
            fprintf(file, "%d", sudoku[i * BOARD_SIZE + j]);

            if (j == BOARD_SIZE - 1)
            {
                fprintf(file, "\n");
            }
        }
    }
    fprintf(file, "\n");
}
