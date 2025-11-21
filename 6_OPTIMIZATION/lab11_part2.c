/*
 * File: lab11_part2.c
 *
 * Based on: Textbook program (page 568) that copies a file to a new file.
 * Changes: This version adds robust command-line argument parsing for:
 *   -h (help), -u (uppercase), -l (lowercase), and two filenames.
 *   It DOES NOT perform the conversion in part 2; it only reports intent.
 *
 * Build:
 *   clang -Wall -Wextra -O2 -o lab11_part2 lab11_part2.c
 *
 * Examples:
 *   ./lab11_part2 -h
 *   ./lab11_part2 -u input.txt output.txt
 *   ./lab11_part2 input.txt output.txt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* Set to 1 to enable verbose debug output, 0 to disable */
#define DEBUG 1

static void print_help(const char* prog) {
    printf("Usage: %s [options] <input_file> <output_file>\n", prog);
    printf("\nOptions:\n");
    printf("  -h        Show this help and exit\n");
    printf("  -u        Request uppercase conversion (reported only in Part 2)\n");
    printf("  -l        Request lowercase conversion (reported only in Part 2)\n");
    printf("\nNotes:\n");
    printf("  - Passing both -u and -l is an error.\n");
    printf("  - If only one filename is provided, the program reports the issue and exits non-zero.\n");
    printf("  - In Part 2, this program DOES NOT perform conversion; it only reports settings.\n");
}

int main(int argc, char* argv[]) {
    bool flagHelp = false;
    bool flagUpper = false;
    bool flagLower = false;
    const char* inputPath = NULL;
    const char* outputPath = NULL;

    /* Scan arguments (order-agnostic). */
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "-h") == 0) {
            flagHelp = true;
        } else if (strcmp(arg, "-u") == 0) {
            flagUpper = true;
        } else if (strcmp(arg, "-l") == 0) {
            flagLower = true;
        } else if (arg[0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", arg);
            return 2;
        } else {
            /* Positional filenames */
            if (!inputPath) {
                inputPath = arg;
            } else if (!outputPath) {
                outputPath = arg;
            } else {
                fprintf(stderr, "Unexpected extra argument: %s\n", arg);
                return 2;
            }
        }
    }

    /* If help was requested, show help and exit 0 regardless of other args. */
    if (flagHelp) {
        print_help(argv[0]);
        return 0;
    }

    /* Validate options */
    if (flagUpper && flagLower) {
        fprintf(stderr, "Error: cannot use -u and -l together.\n");
        return 3;
    }

    if (!inputPath) {
        fprintf(stderr, "Error: missing input filename.\n");
        print_help(argv[0]);
        return 4;
    }
    if (!outputPath) {
        fprintf(stderr, "Error: missing output filename; specify two filenames.\n");
        print_help(argv[0]);
        return 5;
    }

    /* Check that input is readable */
    FILE* in = fopen(inputPath, "rb");
    if (!in) {
        perror("Error opening input file");
        return 6;
    }
    fclose(in);

    /* Part 2: Report what would be done, but do not convert/copy */
    if (DEBUG) {
        printf("help: %s\n", flagHelp ? "true" : "false");
        printf("lowercase: %s\n", flagLower ? "true" : "false");
        printf("uppercase: %s\n", flagUpper ? "true" : "false");
        printf("inputfile: '%s'\n", inputPath);
        printf("outputfile: '%s'\n", outputPath);
    } else {
        printf("OK: ready to process '%s' -> '%s' (conversion planned: %s%s)\n",
               inputPath, outputPath,
               flagUpper ? "upper" : "",
               flagLower ? "lower" : (flagUpper ? "" : "none"));
    }

    return 0;
}






