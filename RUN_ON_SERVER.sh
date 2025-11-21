#!/bin/bash

echo "==========================================="
echo "C LAB - COMPLETE OUTPUT"
echo "==========================================="
echo ""

echo "==========================================="
echo "PART 1A - SWITCH CASE ANALYSIS"
echo "==========================================="
echo "Compilation test (will show error):"
gcc -o part1a part1a.c 2>&1
echo ""
echo "Analysis:"
cat part1a_answer.txt
echo ""
echo ""

echo "==========================================="
echo "PART 1B - STRING FUNCTIONS EXPLANATION"
echo "==========================================="
cat part1b_answer.txt
echo ""
echo ""

echo "==========================================="
echo "PART 2 - STRLEN VS SIZEOF & STRNCMP"
echo "==========================================="
echo "Compiling..."
gcc -o part2 part2.c
echo "Running program:"
./part2
echo ""
echo "Explanation:"
cat part2_answer.txt
echo ""
echo ""

echo "==========================================="
echo "PART 3 - CASE SWAPPING PROGRAM"
echo "==========================================="
echo "Compiling..."
gcc -o part3 part3.c
echo "Test Case 1: AbCdefGuM"
echo "AbCdefGuM" | ./part3
echo ""
echo ""
echo "Test Case 2: HELLO"
echo "HELLO" | ./part3
echo ""
echo ""
echo "Test Case 3: world"
echo "world" | ./part3
echo ""
echo ""

echo "==========================================="
echo "PART 4 - CHEESE SORTING PROGRAM"
echo "==========================================="
echo "Compiling..."
gcc -o part4 part4.c
echo "Running program:"
./part4
echo ""
echo ""

echo "==========================================="
echo "LAB COMPLETE"
echo "==========================================="









