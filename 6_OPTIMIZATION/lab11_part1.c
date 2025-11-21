/*
 * File: lab11_part1.c
 *
 * Source: Program that copies a file to a new file (from textbook, page 568).
 * Note: This implementation is typed from the textbook description. Comments
 * below indicate its original source as requested.
 *
 * Build:
 *   clang -Wall -Wextra -O2 -o lab11_part1 lab11_part1.c
 *
 * Run:
 *   ./lab11_part1 input.txt output.txt
 */

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    const char* inputPath = argv[1];
    const char* outputPath = argv[2];

    FILE* in = fopen(inputPath, "rb");
    if (!in) {
        perror("Error opening input file");
        return 2;
    }

    FILE* out = fopen(outputPath, "wb");
    if (!out) {
        perror("Error opening output file");
        fclose(in);
        return 3;
    }

    /* Textbook-style byte copy loop */
    int ch;
    while ((ch = fgetc(in)) != EOF) {
        if (fputc(ch, out) == EOF) {
            perror("Error writing to output file");
            fclose(in);
            fclose(out);
            return 4;
        }
    }

    if (fclose(in) != 0) {
        perror("Error closing input file");
        /* continue */
    }
    if (fclose(out) != 0) {
        perror("Error closing output file");
        return 5;
    }

    return 0;
}






