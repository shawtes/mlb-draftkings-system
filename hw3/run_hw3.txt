#!/bin/sh

exec > hw3.log 2>&1

date

echo "Building normal binary"
cc -std=c99 -Wall -Wextra -O2 -o hw3_normal hw3.c
if [ $? != 0 ]
then
  echo "  Error -- the last command failed"
  exit 1
else
  echo "  The command worked"
fi

echo "Building debug binary"
cc -std=c99 -Wall -Wextra -O2 -DDEBUG=1 -o hw3_debug hw3.c
if [ $? != 0 ]
then
  echo "  Error -- the last command failed"
  exit 1
else
  echo "  The command worked"
fi

echo "Running debug binary"
./hw3_debug
if [ $? != 0 ]
then
  echo "  Error -- the last command failed"
  exit 1
else
  echo "  The command worked"
fi

echo "Running normal binary"
./hw3_normal
if [ $? != 0 ]
then
  echo "  Error -- the last command failed"
  exit 1
else
  echo "  The command worked"
fi

echo "Output file contents (hw3_output.txt):"
cat ./hw3_output.txt


