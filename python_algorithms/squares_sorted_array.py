#!/usr/bin/env python3
"""
PROBLEM: Squares of a Sorted Array

Given an integer array sorted in non-decreasing order, 
return an array of the squares of each number sorted in non-decreasing order.

KEY INSIGHT:
- The input array is already sorted: [-7, -3, 2, 3, 11]
- After squaring: [49, 9, 4, 9, 121]
- The largest squared values come from the numbers with largest ABSOLUTE values
- These are at the ENDS of the array (most negative or most positive)

APPROACH: Two-Pointer Technique
- Use two pointers: one at start (left), one at end (right)
- Compare absolute values at both ends
- Place the larger square at the END of result array
- Move the pointer that had the larger absolute value
- Fill result array from right to left (largest to smallest)

TIME COMPLEXITY: O(n) - single pass through array
SPACE COMPLEXITY: O(n) - result array
"""

def sortedSquares(nums):
    """
    Returns the squares of a sorted array, also sorted.
    
    Args:
        nums: List of integers sorted in non-decreasing order
    
    Returns:
        List of squared values sorted in non-decreasing order
    """
    n = len(nums)
    result = [0] * n  # Initialize result array with zeros
    
    # Initialize two pointers
    left = 0        # Points to start of array (most negative)
    right = n - 1   # Points to end of array (most positive)
    pos = n - 1     # Fill result from right to left
    
    # Process array from both ends
    while left <= right:
        # Calculate squares of values at both pointers
        left_square = nums[left] ** 2
        right_square = nums[right] ** 2
        
        # Compare and place larger square at current position
        if left_square > right_square:
            # Left value has larger absolute value
            # Example: |-7|^2 = 49 > |3|^2 = 9
            result[pos] = left_square
            left += 1  # Move left pointer inward
        else:
            # Right value has larger or equal absolute value
            # Example: |11|^2 = 121 > |-3|^2 = 9
            result[pos] = right_square
            right -= 1  # Move right pointer inward
        
        pos -= 1  # Move to next position (going left)
    
    return result


def sortedSquares_verbose(nums):
    """
    Same solution with detailed step-by-step output for learning.
    """
    n = len(nums)
    result = [0] * n
    left = 0
    right = n - 1
    pos = n - 1
    
    print(f"\nProcessing array: {nums}")
    print(f"Initial: left={left}, right={right}, pos={pos}\n")
    
    step = 1
    while left <= right:
        left_square = nums[left] ** 2
        right_square = nums[right] ** 2
        
        print(f"Step {step}: Comparing |{nums[left]}|^2={left_square} vs |{nums[right]}|^2={right_square}")
        
        if left_square > right_square:
            result[pos] = left_square
            print(f"         {left_square} > {right_square}, place {left_square} at position {pos}, move left")
            left += 1
        else:
            result[pos] = right_square
            print(f"         {right_square} >= {left_square}, place {right_square} at position {pos}, move right")
            right -= 1
        
        # Show current state of result
        temp_result = ['_' if i > pos else str(result[i]) for i in range(n)]
        print(f"         Result: [{', '.join(temp_result)}]")
        print(f"         left={left}, right={right}, pos={pos-1}\n")
        
        pos -= 1
        step += 1
    
    print(f"Final result: {result}\n")
    return result


# Alternative approaches for comparison
def sortedSquares_naive(nums):
    """
    Naive approach: Square all elements, then sort.
    TIME: O(n log n) due to sorting
    SPACE: O(n)
    """
    # Square each element
    squared = [x ** 2 for x in nums]
    # Sort the result
    squared.sort()
    return squared


def sortedSquares_oneliner(nums):
    """
    Python one-liner using sorted() with key function.
    Same time complexity as naive: O(n log n)
    """
    return sorted([x ** 2 for x in nums])


if __name__ == "__main__":
    # Example 1: nums = [-4,-1,0,3,10]
    print("=" * 60)
    print("EXAMPLE 1")
    print("=" * 60)
    nums1 = [-4, -1, 0, 3, 10]
    print(f"Input:    {nums1}")
    result1 = sortedSquares(nums1)
    print(f"Output:   {result1}")
    print(f"Expected: [0, 1, 9, 16, 100]")
    print(f"Match: {result1 == [0, 1, 9, 16, 100]} ✓\n")
    
    # Example 2: nums = [-7,-3,2,3,11]
    print("=" * 60)
    print("EXAMPLE 2")
    print("=" * 60)
    nums2 = [-7, -3, 2, 3, 11]
    print(f"Input:    {nums2}")
    result2 = sortedSquares(nums2)
    print(f"Output:   {result2}")
    print(f"Expected: [4, 9, 9, 49, 121]")
    print(f"Match: {result2 == [4, 9, 9, 49, 121]} ✓\n")
    
    # Detailed walkthrough of Example 2
    print("=" * 60)
    print("DETAILED WALKTHROUGH OF EXAMPLE 2")
    print("=" * 60)
    sortedSquares_verbose(nums2)
    
    # Compare with naive approach
    print("=" * 60)
    print("COMPARING APPROACHES")
    print("=" * 60)
    test_nums = [-5, -3, -2, -1, 0, 1, 2, 3, 4, 6]
    print(f"Test array: {test_nums}")
    
    result_two_pointer = sortedSquares(test_nums)
    result_naive = sortedSquares_naive(test_nums)
    result_oneliner = sortedSquares_oneliner(test_nums)
    
    print(f"\nTwo-pointer result: {result_two_pointer}")
    print(f"Naive sort result:  {result_naive}")
    print(f"One-liner result:   {result_oneliner}")
    print(f"\nAll methods agree: {result_two_pointer == result_naive == result_oneliner} ✓")
    
    # Performance comparison explanation
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print("Two-Pointer Approach:")
    print("  - Time:  O(n) - single pass")
    print("  - Space: O(n) - result array")
    print("  - Best for this problem!")
    print("\nNaive Sort Approach:")
    print("  - Time:  O(n log n) - due to sorting")
    print("  - Space: O(n) - squared array + sort overhead")
    print("  - Simple but slower")
    print("\nFor n=1,000,000 elements:")
    print("  - Two-pointer: ~1,000,000 operations")
    print("  - Naive sort:  ~20,000,000 operations (20x slower!)")

