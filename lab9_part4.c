#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define BUFFER_SIZE 200

bool printNextLine(char *buffer, char *ptr, int bufferFillLength, int *lineNumber, bool startedLineAlready);

int main(int argc, char *argv[])
{
    FILE *fp;
    char buffer[BUFFER_SIZE];
    int bytesRead;
    int lineNumber = 1;
    bool startedLineAlready = false;
    char *ptr = buffer;
    
    if (argc != 2) {
        printf("usage: lab9_part4 filename\n");
        exit(EXIT_FAILURE);
    }
    
    if ((fp = fopen(argv[1], "r")) == NULL) {
        printf("%s can't be opened\n", argv[1]);
        exit(EXIT_FAILURE);
    }
    
    printf("%s can be opened\n", argv[1]);
    
    while ((bytesRead = fread(buffer, 1, BUFFER_SIZE, fp)) > 0) {
        ptr = buffer;
        
        while (ptr < buffer + bytesRead) {
            startedLineAlready = printNextLine(buffer, ptr, bytesRead, &lineNumber, startedLineAlready);
            
            if (startedLineAlready) {
                while (ptr < buffer + bytesRead && *ptr != '\n') {
                    ptr++;
                }
                if (ptr < buffer + bytesRead) {
                    ptr++; // Skip the newline character
                }
            } else {
                while (ptr < buffer + bytesRead && *ptr != '\n') {
                    ptr++;
                }
                if (ptr < buffer + bytesRead) {
                    ptr++; // Skip the newline character
                }
            }
        }
    }
    
    fclose(fp);
    return 0;
}

bool printNextLine(char *buffer, char *ptr, int bufferFillLength, int *lineNumber, bool startedLineAlready)
{
    char *lineStart = ptr;
    char *lineEnd = ptr;
    char *p;
    bool foundNewline = false;
    
    while (lineEnd < buffer + bufferFillLength && *lineEnd != '\n') {
        lineEnd++;
    }
    
    if (lineEnd < buffer + bufferFillLength) {
        foundNewline = true;
    }
    
    if (!foundNewline) {
        if (!startedLineAlready) {
            printf("%d \"", *lineNumber);
            (*lineNumber)++;
        }
        
        for (p = lineStart; p < lineEnd; p++) {
            // Check if character is printable text
            unsigned char ch = (unsigned char)*p;
            if (ch >= 32 && ch <= 127) {
                // Printable ASCII characters (32-127)
                printf("%c", ch);
            } else if (ch == 9 || ch == 10) {
                // Tab (9) and Line Feed (10) are allowed
                printf("%c", ch);
            } else {
                // Non-text characters - print comma instead
                printf(",");
            }
        }
        
        return true; // Need to read more from file
    } else {
        // We found a complete line
        if (!startedLineAlready) {
            printf("%d \"", *lineNumber);
            (*lineNumber)++;
        }
        
        for (p = lineStart; p < lineEnd; p++) {
            // Check if character is printable text
            unsigned char ch = (unsigned char)*p;
            if (ch >= 32 && ch <= 127) {
                // Printable ASCII characters (32-127)
                printf("%c", ch);
            } else if (ch == 9 || ch == 10) {
                // Tab (9) and Line Feed (10) are allowed
                printf("%c", ch);
            } else {
                // Non-text characters - print comma instead
                printf(",");
            }
        }
        
        printf("\"\n");
        return false; // Line is complete
    }
}
