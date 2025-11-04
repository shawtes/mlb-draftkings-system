#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/param.h>
#include <limits.h>

#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 4096
#endif

static char *next_space_or_tab(const char *s) {
    if (s == NULL) return NULL;
    const char *p = s;
    while (*p != '\0') {
        if (*p == ' ' || *p == '\t') return (char *)p;
        p++;
    }
    return (char *)p;
}

static void trim_trailing_newline(char *s) {
    if (!s) return;
    size_t n = strlen(s);
    if (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r')) s[n - 1] = '\0';
}

static void run_debug_tests(void) {
    const char *tests[] = {
        "123 abc",
        "123\tabc",
        "123abc",
        " 123abc",
        "\tabc",
        "abc def ghi",
        "",
    };
    size_t num = sizeof(tests) / sizeof(tests[0]);
    for (size_t i = 0; i < num; i++) {
        const char *t = tests[i];
        char *p = next_space_or_tab(t);
        printf("TEST %zu: input=\"%s\" remainder=\"%s\"\n", i + 1, t, p ? p : "(null)");
    }
}

typedef struct {
    long long size;
    char name[BUFFER_SIZE];
    int valid;
} Entry;

static void maybe_print(FILE *f, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    if (f) {
        va_start(ap, fmt);
        vfprintf(f, fmt, ap);
        va_end(ap);
        fflush(f);
    }
    fflush(stdout);
}

int main(void) {
    if (DEBUG) {
        run_debug_tests();
        return 0;
    }

    char initial_dir[MAXPATHLEN];
    if (getcwd(initial_dir, sizeof(initial_dir)) == NULL) {
        fprintf(stderr, "getcwd failed\n");
        return 1;
    }

    char out_path[MAXPATHLEN + 64];
    snprintf(out_path, sizeof(out_path), "%s/%s", initial_dir, "hw3_output.txt");
    FILE *outf = fopen(out_path, "w");
    if (!outf) {
        fprintf(stderr, "failed to open output file: %s\n", out_path);
        return 1;
    }

    maybe_print(outf, "Initial directory is %s\n", initial_dir);

    if (chdir("/usr/lib") != 0) {
        maybe_print(outf, "chdir to /usr/lib failed\n");
        fclose(outf);
        return 1;
    }

    FILE *proc = popen("du -k -d 1 .", "r");
    if (!proc) {
        maybe_print(outf, "popen du failed\n");
        fclose(outf);
        return 1;
    }

    Entry top[3];
    for (int i = 0; i < 3; i++) {
        top[i].size = -1;
        top[i].name[0] = '\0';
        top[i].valid = 0;
    }

    char line[BUFFER_SIZE];
    while (fgets(line, sizeof(line), proc) != NULL) {
        trim_trailing_newline(line);
        if (line[0] == '\0') continue;

        char *endptr = NULL;
        long long size = strtoll(line, &endptr, 10);
        if (endptr == line) continue;

        while (*endptr == ' ' || *endptr == '\t') endptr++;
        if (*endptr == '\0') continue;

        char namebuf[BUFFER_SIZE];
        strncpy(namebuf, endptr, sizeof(namebuf) - 1);
        namebuf[sizeof(namebuf) - 1] = '\0';

        int idx = -1;
        for (int i = 0; i < 3; i++) {
            if (!top[i].valid || size > top[i].size) {
                idx = i;
                break;
            }
        }
        if (idx >= 0) {
            for (int j = 2; j > idx; j--) top[j] = top[j - 1];
            top[idx].size = size;
            strncpy(top[idx].name, namebuf, sizeof(top[idx].name) - 1);
            top[idx].name[sizeof(top[idx].name) - 1] = '\0';
            top[idx].valid = 1;
        }
    }

    pclose(proc);

    int count = 0;
    for (int i = 0; i < 3; i++) if (top[i].valid) count++;

    for (int i = 0; i < count; i++) {
        maybe_print(outf, "%lld %s\n", top[i].size, top[i].name);
    }

    if (count >= 1) {
        long long rest = top[0].size;
        for (int i = 1; i < count; i++) rest -= top[i].size;
        maybe_print(outf, "The rest use %lld\n", rest);
    }

    fclose(outf);
    return 0;
}


