# ============================================================================
# PROBLEM: Squares of a Sorted Array
# ============================================================================
# "Given an integer array nums sorted in non-decreasing order, 
#  return an array of the squares of each number sorted in non-decreasing order.
# 
# Example 1:
#   Input: nums = [-4,-1,0,3,10]
#   Output: [0,1,9,16,100]
#   Explanation: After squaring, the array becomes [16,1,0,9,100].
#                After sorting, it becomes [0,1,9,16,100].
# 
# Example 2:
#   Input: nums = [-7,-3,2,3,11]
#   Output: [4,9,9,49,121]
# 
# Constraints:
#   1 <= nums.length <= 104
#   -104 <= nums[i] <= 104
#   nums is sorted in non-decreasing order.
# 
# Follow up: Squaring each element and sorting the new array is very trivial, 
#            could you find an O(n) solution using a different approach?"
# ============================================================================


# ============================================================================
# SOLUTION 1: Easy Approach - Square then Sort
# Time: O(n log n) because of sorting
# Space: O(n) for the result array
# ============================================================================

def sortedSquares_easy(nums):
    """
    Simplest approach: Square everything, then sort.
    
    Why this works:
    - We square each number (negative numbers become positive)
    - Then we sort the result
    
    Example: [-4, -1, 0, 3, 10]
    Step 1: Square everything → [16, 1, 0, 9, 100]
    Step 2: Sort → [0, 1, 9, 16, 100]
    """
    # Step 1: Square each number
    result = []
    for num in nums:
        squared = num * num  # or num ** 2
        result.append(squared)
    
    # Step 2: Sort the squared numbers
    result.sort()
    
    return result


# ============================================================================
# SOLUTION 2: Optimal Approach - Two Pointers
# Time: O(n) - only go through array once!
# Space: O(n) for the result array
# ============================================================================

def sortedSquares_optimal(nums):
    """
    Smart approach using Two Pointers (THIS IS THE O(n) SOLUTION!)
    
    KEY INSIGHT:
    - The array is ALREADY sorted
    - The LARGEST squares come from either:
      1. Very negative numbers (far left)  → like -10 squared = 100
      2. Very positive numbers (far right) → like 10 squared = 100
    
    STRATEGY:
    - Use two pointers: one at start (left), one at end (right)
    - Compare the squares of both
    - The BIGGER square goes at the END of our result
    - Move the pointer that gave us the bigger square
    - Fill result array from RIGHT to LEFT
    
    Example walkthrough: [-4, -1, 0, 3, 10]
    
    left = 0 (pointing to -4)    right = 4 (pointing to 10)
    result = [_, _, _, _, _]     position = 4 (start from end)
    
    Step 1: Compare |-4|² vs |10|²  →  16 vs 100
            100 is bigger! Put it at position 4
            result = [_, _, _, _, 100]
            Move right pointer left (right = 3)
            position = 3
    
    Step 2: Compare |-4|² vs |3|²  →  16 vs 9
            16 is bigger! Put it at position 3
            result = [_, _, _, 16, 100]
            Move left pointer right (left = 1)
            position = 2
    
    Step 3: Compare |-1|² vs |3|²  →  1 vs 9
            9 is bigger! Put it at position 2
            result = [_, _, 9, 16, 100]
            Move right pointer left (right = 2)
            position = 1
    
    Step 4: Compare |-1|² vs |0|²  →  1 vs 0
            1 is bigger! Put it at position 1
            result = [_, 1, 9, 16, 100]
            Move left pointer right (left = 2)
            position = 0
    
    Step 5: Only one element left (0)
            result = [0, 1, 9, 16, 100]
            Done!
    """
    
    n = len(nums)
    result = [0] * n  # Create array of same length, filled with zeros
    
    # Two pointers: start and end of the array
    left = 0
    right = n - 1
    
    # Position to fill in result (start from the end)
    position = n - 1
    
    # Keep going until pointers meet
    while left <= right:
        # Get the absolute values (to compare magnitudes)
        left_square = nums[left] * nums[left]
        right_square = nums[right] * nums[right]
        
        # The bigger square goes in our current position
        if left_square > right_square:
            result[position] = left_square
            left += 1  # Move left pointer to the right
        else:
            result[position] = right_square
            right -= 1  # Move right pointer to the left
        
        # Move to next position (going backwards)
        position -= 1
    
    return result


# ============================================================================
# TEST CASES
# ============================================================================

if __name__ == "__main__":
    # Test Case 1
    nums1 = [-4, -1, 0, 3, 10]
    print("Test 1:")
    print(f"  Input:  {nums1}")
    print(f"  Easy:   {sortedSquares_easy(nums1)}")
    print(f"  Optimal: {sortedSquares_optimal(nums1)}")
    print(f"  Expected: [0, 1, 9, 16, 100]")
    print()
    
    # Test Case 2
    nums2 = [-7, -3, 2, 3, 11]
    print("Test 2:")
    print(f"  Input:  {nums2}")
    print(f"  Easy:   {sortedSquares_easy(nums2)}")
    print(f"  Optimal: {sortedSquares_optimal(nums2)}")
    print(f"  Expected: [4, 9, 9, 49, 121]")
    print()
    
    # Test Case 3: All negative
    nums3 = [-5, -3, -2, -1]
    print("Test 3 (all negative):")
    print(f"  Input:  {nums3}")
    print(f"  Easy:   {sortedSquares_easy(nums3)}")
    print(f"  Optimal: {sortedSquares_optimal(nums3)}")
    print(f"  Expected: [1, 4, 9, 25]")
    print()
    
    # Test Case 4: All positive
    nums4 = [1, 2, 3, 4, 5]
    print("Test 4 (all positive):")
    print(f"  Input:  {nums4}")
    print(f"  Easy:   {sortedSquares_easy(nums4)}")
    print(f"  Optimal: {sortedSquares_optimal(nums4)}")
    print(f"  Expected: [1, 4, 9, 16, 25]")
    print()


# ============================================================================
# WHY THE OPTIMAL SOLUTION IS BETTER
# ============================================================================
"""
Easy Solution:
- Time: O(n log n) because sorting takes n log n time
- If array has 1000 elements, ~10,000 operations

Optimal Solution:
- Time: O(n) because we only look at each element once
- If array has 1000 elements, only 1000 operations

For large arrays, optimal is MUCH faster!

Example:
- Array size = 100,000 elements
- Easy:    100,000 * log(100,000) ≈ 1,660,000 operations
- Optimal: 100,000 operations

The optimal solution is ~16x faster! 🚀
"""