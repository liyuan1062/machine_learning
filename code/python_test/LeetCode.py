# coding=utf-8


class LeetCode(object):
    def __init__(self, nums, target):
        # print (self.twoSum_2(nums, target))
        pass
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        sorted_nums = sorted(nums)

        original_index = [i[0] for i in sorted(enumerate(nums), key=lambda x:x[1])]
        for i, v in enumerate(sorted_nums):
            for j in range(i+1, len(sorted_nums)):
                if v+sorted_nums[j] == target:
                    return [original_index[i], original_index[j]]
                elif v+sorted_nums[j] > target:
                    break

        raise Exception("can't find a solution!")

    def twoSum_2(self, nums, target):
        d = dict()
        for i, v in enumerate(nums):
            m = target - v
            if m in d:
                return [d[m], i]
            else:
                d[v] = i
    @classmethod
    def addTwoNumbers(cls, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        max_len = max(len(l1), len(l2))
        if len(l1) < len(l2):
            [l1.append(0) for i in range(max_len - len(l1))]
        else:
            [l2.append(0) for i in range(max_len - len(l2))]

        result = []
        up = 0
        for i in range(max_len):
            s = l1[i] + l2[i] + up
            if s < 10:
                up = 0
                result.append(s)
            else:
                up = 1
                result.append(s - 10)
        if up:
            result.append(1)
        return result

    # class ListNode(object):
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None
    # @classmethod
    # def addTwoNumbers_2(self, l1, l2):
    #     h = cur = ListNode(0)
    #     carry = 0
    #     while l1 or l2 or carry:
    #         if l1:
    #             carry += l1.val
    #             l1 = l1.next
    #         if l2:
    #             carry += l2.val
    #             l2 = l2.next
    #         cur.next = ListNode(carry % 10)
    #         cur = cur.next
    #         carry = carry // 10
    #     return h.next

    @staticmethod
    def lengthOfLongestSubstring(s):
        """
        :type s: str
        :rtype: int
        """
        d = {}
        start = 0
        maxLen = 0
        for i, v in enumerate(s):
            if v in d and start <= d[v]:
                start = d[v] + 1
            else:
                maxLen = max(maxLen, i - start + 1)
            d[v] = i
        return maxLen

    @staticmethod
    def binarySearch(nums, val, start=-1, end=-1):
        if start == -1 or end == -1:
            start, end = 0, len(nums)-1
        print("start :{}, end: {}".format(start, end))
        if start > end:
            raise Exception("can't find target val {} in list".format(val))
        index = start + (end - start) // 2
        if nums[index] == val:
            return index
        elif nums[index] > val:
            index = LeetCode.binarySearch(nums, val, start, index-1)
        else:
            index = LeetCode.binarySearch(nums, val, index+1, end)
        return index

    @staticmethod
    def binarySearch_2(nums, val):
        start, end = 0, len(nums) - 1
        found = False
        res = 0
        while start <= end and not found:
            print("start :{}, end: {}".format(start, end))
            midIndex = (start + end) // 2
            if nums[midIndex] == val:
                found = True
                res = midIndex
            elif nums[midIndex] > val:
                end = midIndex - 1
            else:
                start = midIndex + 1
        if found:
            return res
        else:
            raise Exception("can't find target val {} in list".format(val))

    @staticmethod
    def kthSmallest(nums1, nums2, k):
        '''
        O(n+k) time complexity
        :param nums1:
        :param nums2:
        :param k:
        :return:
        '''
        if not nums1:
            return nums2[k]
        if not nums2:
            return nums1[k]

        if k > len(nums1) + len(nums2):
            raise Exception("no kth element in two lists")
        i, j = 0, 0
        kk = k + 1
        item = 0
        while kk > 0:
            if i >= len(nums1):
                return nums2[k - i]
            elif j >= len(nums2):
                return nums1[k - j]
            elif nums1[i] > nums2[j]:
                item = nums2[j]
                j += 1
            elif nums1[i] <= nums2[j]:
                item = nums1[i]
                i += 1
            kk -= 1
        return item

    @staticmethod
    def kthSmallest_2(nums1, nums2, k):
        '''
        :param nums1:
        :param nums2:
        :param k:
        :return:
        '''
        if not nums1:
            return nums2[k]
        if not nums2:
            return nums1[k]
        ia = len(nums1)//2
        ib = len(nums2)//2
        ma = nums1[ia]
        mb = nums2[ib]
        if ia + ib < k:
            if ma < mb:
                return LeetCode.kthSmallest_2(nums1[ia+1:], nums2, k-ia-1)
            else:
                return LeetCode.kthSmallest_2(nums1, nums2[ib+1:], k-ib-1)
        else:
            if ma < mb:
                return LeetCode.kthSmallest_2(nums1, nums2[:ib], k - ib)
            else:
                return LeetCode.kthSmallest_2(nums1[:ia], nums2, k - ia)



    @staticmethod
    def findMedianSortedArrays(nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        l1 = len(nums1)
        l2 = len(nums2)
        if (l1 + l2) % 2 == 1:
            return float(LeetCode.kthSmallest_2(nums1, nums2, (l1 + l2) // 2))
        else:
            i = LeetCode.kthSmallest_2(nums1, nums2, (l1+l2)//2)
            j = LeetCode.kthSmallest_2(nums1, nums2, (l1+l2)//2 - 1)
            return i+j/2.0

    @staticmethod
    def findLongestPalindromic(s):
        d = {}
        for i, v in enumerate(s):
            if v in d:
                d[v].append(i)
            else:
                d[v] = [i]
        res = 0
        startIndex = 0
        for start, v in enumerate(s):
            indexes = d[v]
            if len(d) == 1:
                res = len(s)
                startIndex = 0
            if len(indexes) < 2 :
                continue
            for end in indexes[::-1]:
                if end > start and LeetCode.isPalindromic(s, start, end) and (end - start + 1) > res:
                    res = end - start + 1
                    startIndex = start
        if res > 0:
            return s[startIndex:startIndex+res]
        else:
            return s[0]
    @staticmethod
    def isPalindromic(str, start, end):
        i, j = start, end
        while i <= (end + start)//2:
            if str[i] != str[j]:
                return False
            i += 1
            j -= 1
        return True

    @staticmethod
    def expandAroundCenter(s, left, right):
        res = 0
        while(left >= 0) and (right < len(s)) and (s[left] == s[right]):
            res = right - left + 1
            left -= 1
            right += 1
        return res

    @staticmethod
    def findLongestPalindromic_2(s):
        res = 0
        start = 0
        for i in range(len(s)):
            l1 = LeetCode.expandAroundCenter(s, i, i)
            l2 = LeetCode.expandAroundCenter(s, i, i+1)
            if l1 >= l2 and l1 > res:
                res = l1
                start = i - res // 2
            if l1 < l2 and l2 > res:
                res = l2
                start = i - (res-1)//2
        return s[start:start + res]

if __name__ == "__main__":
    # p1 = LeetCode([3,2,4], 6)

    # p1 = print(LeetCode.addTwoNumbers([2,4,5], [5,6,4]))

    # p = print(LeetCode.lengthOfLongestSubstring("tmmzuxt"))
    a = [2,4,6,7,9,11]
    b = [1,3,5,6,7]
    c = [1,2]
    d = [3,4]
    # print(LeetCode.binarySearch_2(a, 11))
    # print(LeetCode.findMedianSortedArrays(c, d))
    s = "abababab"
    print(LeetCode.findLongestPalindromic_2(s))
