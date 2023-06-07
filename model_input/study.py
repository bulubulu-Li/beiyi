class Solution(object):
    def heap_sort(self,nums):
        i,l = 0,len(nums)
        self.nums = nums
        for i in range(l//2-1,-1,-1):
            self.bulid_heap(i,l-1)
        for j in range(l-1,-1,-1):
            self.nums[0],self.nums[j] = self.nums[j],self.nums[0]
            self.bulid_heap(0,j-1)
        return self.nums


    def bulid_heap(self,i,l):
        left = 2*i+1
        right = 2*i+2
        large_index = i
        if left<=l and self.nums[left]>self.nums[large_index]:
            large_index = left
        if right<=l and self.nums[right]>self.nums[large_index]:
            large_index = right
        if large_index!=i:
            self.nums[i],self.nums[large_index] = self.nums[large_index],self.nums[i]
            self.bulid_heap(large_index,l)
p = Solution()
print(p.heap_sort([2,4,1,5,9,7]))