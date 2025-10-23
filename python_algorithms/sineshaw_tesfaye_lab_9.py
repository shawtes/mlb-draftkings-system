

class MinHeap:
   
    def __init__(self):
        self.heap = []
    
    def insert(self, value):
        self.heap.append(value)
        self._bubble_up(len(self.heap) - 1)
    
    def remove(self):
        if self.is_empty():
            raise IndexError("Cannot remove from an empty heap")
        
        min_value = self.heap[0]
        
        last_element = self.heap.pop()
        
        if not self.is_empty():
            self.heap[0] = last_element
            self._bubble_down(0)
        
        return min_value
    
    def _bubble_up(self, index):
        while index > 0:
            parent_index = self._parent(index)         
            if self.heap[index] < self.heap[parent_index]:
                self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
                index = parent_index
            else:
                break
    
    def _bubble_down(self, index):
        while True:
            left_child_index = self._left_child(index)
            right_child_index = self._right_child(index)
            smallest = index
            
            if (left_child_index < len(self.heap) and 
                self.heap[left_child_index] < self.heap[smallest]):
                smallest = left_child_index            
            if (right_child_index < len(self.heap) and 
                self.heap[right_child_index] < self.heap[smallest]):
                smallest = right_child_index            
            if smallest != index:
                self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
                index = smallest
            else:
                break
    
    def _parent(self, index):
        return (index - 1) // 2
    
    def _left_child(self, index):
        return 2 * index + 1
    
    def _right_child(self, index):
        return 2 * index + 2
    
    def peek(self):

        if self.is_empty():
            raise IndexError("Cannot peek at an empty heap")
        return self.heap[0]
    
    def size(self):
        return len(self.heap)
    
    def is_empty(self):
        return len(self.heap) == 0
    
    def __str__(self):
        return str(self.heap)
    
    def __repr__(self):
        return f"MinHeap({self.heap})"


def test_min_heap():
    print("Min Heap Test\n")
    
    heap = MinHeap()
    print(f"Created empty heap: {heap}")
    print(f"Is empty? {heap.is_empty()}\n")
    
    values = [5, 3, 7, 1, 9, 2, 8, 4, 6]
    print(f"Inserting values: {values}")
    for value in values:
        heap.insert(value)
        print(f"After inserting {value}: {heap}")
    
    print(f"\nHeap size: {heap.size()}")
    print(f"Minimum value (peek): {heap.peek()}\n")
    
    print("Removing all elements in sorted order:")
    removed_values = []
    while not heap.is_empty():
        min_val = heap.remove()
        removed_values.append(min_val)
        print(f"Removed: {min_val}, Heap: {heap}")
    
    print(f"\nRemoved values in order: {removed_values}")
    print(f"Is empty? {heap.is_empty()}")
    
    assert removed_values == sorted(values), "Min heap is not working correctly!"
    print("\n✓ All tests passed! Min heap is working correctly.")
    
    print("\nTesting Edge Cases\n")
    
    try:
        heap.remove()
    except IndexError as e:
        print(f"✓ Correctly raised error when removing from empty heap: {e}")
    
    try:
        heap.peek()
    except IndexError as e:
        print(f"✓ Correctly raised error when peeking at empty heap: {e}")
    
    heap.insert(42)
    print(f"\nInserted single element: {heap}")
    print(f"Removed: {heap.remove()}")
    print(f"Empty after removal? {heap.is_empty()}")
    
    print("\nTesting Duplicate Values\n")
    duplicate_values = [5, 2, 8, 2, 9, 2, 1]
    print(f"Inserting values with duplicates: {duplicate_values}")
    for value in duplicate_values:
        heap.insert(value)
    print(f"Heap: {heap}")
    
    print("\nRemoving all elements:")
    duplicate_removed = []
    while not heap.is_empty():
        duplicate_removed.append(heap.remove())
    print(f"Removed in order: {duplicate_removed}")
    print(f"Sorted original: {sorted(duplicate_values)}")
    assert duplicate_removed == sorted(duplicate_values), "Duplicate handling failed!"
    print("✓ Duplicates handled correctly!")


if __name__ == "__main__":
    test_min_heap()

