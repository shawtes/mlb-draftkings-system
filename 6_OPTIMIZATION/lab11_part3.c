/*
 * File: lab11_part3.c
 *
 * Based on: Textbook program (page 568) that copies a file to a new file.
 * Changes: Extends Part 2 by implementing upper/lower-case conversion of
 * alphabetic characters when -u or -l is specified.
 *
 * Build:
 *   clang -Wall -Wextra -O2 -o lab11_part3 lab11_part3.c
 *
 * Examples:
 *   ./lab11_part3 -u in.txt out.txt
 *   ./lab11_part3 -l in.txt out.txt
 *   ./lab11_part3 in.txt out.txt  (no conversion, just copy)
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
    printf("  -u        Convert alphabetic characters to UPPER-CASE\n");
    printf("  -l        Convert alphabetic characters to lower-case\n");
    printf("\nNotes:\n");
    printf("  - Passing both -u and -l is an error.\n");
    printf("  - If only one filename is provided, the program reports the issue and exits non-zero.\n");
}

static int convert_char(int ch, bool toUpper, bool toLower) {
    /* From assignment:
       if ((ch >= 'a') && (ch <= 'z')) { if (convertToUpper) ch = ch - 'a' + 'A'; }
       else if ((ch >= 'A') && (ch <= 'Z')) { if (convertToLower) ch = ch - 'A' + 'a'; }
    */
    if ((ch >= 'a') && (ch <= 'z')) {
        if (toUpper) {
            ch = ch - 'a' + 'A';
        }
    } else if ((ch >= 'A') && (ch <= 'Z')) {
        if (toLower) {
            ch = ch - 'A' + 'a';
        }
    }
    return ch;
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

    FILE* in = fopen(inputPath, "rb");
    if (!in) {
        perror("Error opening input file");
        return 6;
    }
    FILE* out = fopen(outputPath, "wb");
    if (!out) {
        perror("Error opening output file");
        fclose(in);
        return 7;
    }

    if (DEBUG) {
        printf("help: %s\n", flagHelp ? "true" : "false");
        printf("lowercase: %s\n", flagLower ? "true" : "false");
        printf("uppercase: %s\n", flagUpper ? "true" : "false");
        printf("inputfile: '%s'\n", inputPath);
        printf("outputfile: '%s'\n", outputPath);
    }

    /* Copy with optional conversion */
    int ch;
    while ((ch = fgetc(in)) != EOF) {
        int outCh = convert_char(ch, flagUpper, flagLower);
        if (fputc(outCh, out) == EOF) {
            perror("Error writing to output file");
            fclose(in);
            fclose(out);
            return 8;
        }
    }

    if (fclose(in) != 0) {
        perror("Error closing input file");
        /* continue */
    }
    if (fclose(out) != 0) {
        perror("Error closing output file");
        return 9;
    }

    return 0;
}






