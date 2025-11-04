import heapq

class DataStream:
    def __init__(self):
        self.min_heap = []  
        self.max_heap = []  

    def add(self, num: int) -> None:
        if not self.max_heap or num <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
        else:
            heapq.heappush(self.min_heap, num)

        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap) + 1:
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def get_median(self) -> float:
        if len(self.max_heap) == len(self.min_heap):
            if not self.max_heap:  
                return 0.0
            return (-self.max_heap[0] + self.min_heap[0]) / 2.0
        return float(-self.max_heap[0]) if len(self.max_heap) > len(self.min_heap) else float(self.min_heap[0])


# Test
stream = DataStream()
data = [4, 1, 3, 9, 2, 11, 14, 5]
medians = []
for x in data:
    stream.add(x)
    medians.append(stream.get_median())

print(medians)
