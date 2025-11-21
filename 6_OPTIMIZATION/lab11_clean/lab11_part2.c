#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

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
}

int main(int argc, char* argv[]) {
    bool flagHelp = false;
    bool flagUpper = false;
    bool flagLower = false;
    const char* inputPath = NULL;
    const char* outputPath = NULL;

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

    if (flagHelp) {
        print_help(argv[0]);
        return 0;
    }

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
    fclose(in);

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






