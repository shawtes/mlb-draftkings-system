#!/bin/bash

echo "========================================="
echo "PART 1A - SWITCH CASE ANALYSIS"
echo "========================================="
echo "Attempting to compile part1a.c..."
gcc -o part1a part1a.c 2>&1
echo ""
echo "Analysis:"
cat part1a_answer.txt
echo ""

echo "========================================="
echo "PART 1B - STRING FUNCTIONS"
echo "========================================="
cat part1b_answer.txt
echo ""

echo "========================================="
echo "PART 2 - STRLEN VS SIZEOF"
echo "========================================="
echo "Compiling part2.c..."
gcc -o part2 part2.c
echo "Running part2..."
./part2
echo ""
echo "Explanation:"
cat part2_answer.txt
echo ""

echo "========================================="
echo "PART 3 - CASE SWAPPING"
echo "========================================="
echo "Compiling part3.c..."
gcc -o part3 part3.c
echo "Program compiled. Run './part3' and enter 'AbCdefGuM' to test"
echo ""

echo "========================================="
echo "PART 4 - CHEESE SORTING"
echo "========================================="
echo "Compiling part4.c..."
gcc -o part4 part4.c
echo "Running part4..."
./part4
echo ""

echo "========================================="
echo "ALL TASKS COMPLETE"
echo "========================================="









