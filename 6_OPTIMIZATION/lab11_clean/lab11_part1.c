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
    }
    if (fclose(out) != 0) {
        perror("Error closing output file");
        return 5;
    }

    return 0;
}






