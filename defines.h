#define BOARD_SIZE 9
#define BOX_SIZE 3
#define THREADS 512
#define BLOCKS 1024
#define MAX_BOARDS 1000000
#define ITERATIONS_COUNT 20
#define ERR(source) (perror(source),                                 \
                     fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), \
                     exit(EXIT_FAILURE))
