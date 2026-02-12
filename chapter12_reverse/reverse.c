/* REVERSE.C reusable function derived from Chapter 12 (p. 389) */

#include <string.h>

#include "reverse.h"

void reverse(const char *before, char *after)
{
    int i;
    int j;
    int len;

    len = (int)strlen(before);

    for (i = 0, j = len - 1; j >= 0; j--, i++) /* Reverse loop */
    {
        after[i] = before[j];
    }

    after[len] = '\0'; /* NULL terminate reversed string */
}
