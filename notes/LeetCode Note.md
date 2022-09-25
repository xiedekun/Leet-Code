## 二叉树

**层序数组转树**

```
def buildTree(nums, i):
    
    if i >= len(nums):
        return 
    
    if not nums[i]:
        return

    root = TreeNode(nums[i])
        
    root.left = buildTree(nums, 2*i+1)
    root.right = buildTree(nums, 2*i+2)
    
    return root
```

### 【遍历】

##### 1.前序遍历
https://leetcode.cn/problems/binary-tree-preorder-traversal/

>递归

**思路**：先存根节点，在按照左子树和右子树递归。

```
    res.append(root.val)
    if root.left:
        res.extend(preorderTraversal(root.left))
    if root.right:
        res.extend(preorderTraversal(root.right))
    return res
```


>迭代

**思路**：用栈，**循环前先把根节点加入**， 按遍历反顺序放入，弹出（先进后出）。

```
stack = [root]

    while stack:
        node = stack.pop()
        res.append(node.val)
        
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return res
```


##### 2.中序遍历
https://leetcode.cn/problems/binary-tree-inorder-traversal/
>递归

**思路**：只需要按照遍历位置放置就可以。
```
    if root.left:
        res.extend(inorderTraversal(root.left))

    res.append(root.val)
    
    if root.right:
        res.extend(inorderTraversal(root.right))
    return res
```
>遍历

**思路**：循环外**空栈**加node，先把左边压完，弹出记录再压右边。
```
    stack = []
    node = root
    
    while node or stack:
        if node:
            stack.append(node)
            node = node.left
        else:
            node = stack.pop()
            res.append(node.val)
            node = node.right
    return res
```

##### 3.后续遍历
https://leetcode.cn/problems/binary-tree-postorder-traversal/
>递归

**思路**：只需要按照遍历位置放置就可以。
```
    if root.left:
        res.extend(postorderTraversal(root.left))
    
    if root.right:
        res.extend(postorderTraversal(root.right))

    res.append(root.val)
```
>迭代

**思路**：对比前序遍历，左右子树反着压入栈，最后逆着输出结果。
```
stack = [root]

    while stack:
        node = stack.pop()
        res.append(node.val)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

            
    return res[::-1]
```

##### 4.层序遍历
https://leetcode.cn/problems/binary-tree-level-order-traversal/

**思路**：利用队列先进先出，广度优先搜索。循环前根节点先入队。

```
    queue = deque([root])
    
    while queue:
        size = len(queue)
        res_ = []
        while size > 0:
            
            node = queue.popleft()

            res_.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            size -= 1
        res.append(res_)
    return res
```

### 【构建树】

https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/

**思路**：根据前序或者后序找到根节点，和其在中序遍历中的位置k。从而知道左子树的长度k-1和右子树的长度，传入前/后序遍历递归。

```
    root = TreeNode(preorder[0])
    k = inorder.index(preorder[0])
    
    root.left = buildTree(preorder[1:k+1], inorder[0:k])
    root.right = buildTree(preorder[k+1:], inorder[k+1:])
    return root
```

### 【不同的二叉搜索树】

https://leetcode.cn/problems/unique-binary-search-trees/

**思路**：动态规划

dp[i] 节点数为i的二叉搜索树个数

    1 => 1
    2 => 2
    3 => 5
取x为根节点
左子树j，右子树i-j-1

i-j-1>=0 => i-1>=j

以x为根节点的数量有 dp[j] * dp[i-j-1]

dp[i] = sum(dp[j]*dp[i-j-1])

子树可以为0，所以初始化要注意dp[0] = 1

dp[1] = 1


```
def numTrees(n):
    """
    :type n: int
    :rtype: int
    """
    dp = [0]*(n+1)
    dp[0] = 1
    dp[1] = 1

    for i in range(2,n+1):
        for j in range(i):
            dp[i] +=dp[j]*dp[i-j-1]


    return dp[-1]
```
### 【不同的二叉搜索树 II】

https://leetcode.cn/problems/unique-binary-search-trees-ii/

**思路**： 递归
搜索二叉树：左小右大

循环递归：3次循环，根节点；左子树nums[:i]；右子树nums[i+1:]；子树递归，node加入res

```
def generateTrees(n):
    """
    :type n: int
    :rtype: List[TreeNode]
    """
    if n == 0:
        return []

    nums = [i for i in range(1,n+1)]
    
    def helper(nums):

        if not nums:
            return [None]
        
        res = []
        for i, num in enumerate(nums):
            for l in helper(nums[:i]):
                for r in helper(nums[i+1:]):
                    node = TreeNode(num)
                    node.left = l
                    node.right = r
                    res.append(node)
        return res

    return helper(nums)
```

### 【 恢复二叉搜索树】

https://leetcode.cn/problems/recover-binary-search-tree

**思路**：中序遍历以搜寻两个异常节点：node1肯定是大于后面不符合，node2肯定是小于前面不符合；需要用pre变量记录前一个节点；最后交换node1和node2的值。

```
def recoverTree(root):
    """
    :type root: TreeNode
    :rtype: None Do not return anything, modify root in-place instead.
    """
    pre = node1 = node2 = None

    def inorder(node):

        nonlocal pre, node1, node2
        
        if not node:
            return 

        inorder(node.left)

        if pre!=None and pre.val>node.val:
            if not node1:
                node1 = pre #第一个点肯定是大于后面的点不符合
            node2 = node # 第二个点一定是小于前面的点不符合

        pre = node

        inorder(node.right)

    inorder(root)
    node1.val, node2.val = node2.val, node1.val    
```
### 【翻转二叉树】

https://leetcode.cn/problems/invert-binary-tree/

**思路**：层序遍历，左右子树交换。

```
    queue = []
    
    queue.append(root)
    while queue:
        node = queue.pop(0)
        ls = node.left
        node.left = node.right
        node.right = ls
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return root
```
### 【判断平衡二叉树】

https://leetcode.cn/problems/balanced-binary-tree/

**思路**：递归先计算当前根节点最大深度；递归比较左右子树最大深度差是否小于1并且递归查看左子树和右子树是否为平衡树

```
def isBalanced(root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    def height(root):

        if not root:
            return 0
        h = max(height(root.left), height(root.right))+1
        return h
    
    if not root:
        return True
    
    return abs(height(root.left) - height(root.right))<=1 and isBalanced(root.left) and isBalanced(root.right)
```

### 【路径总和】

https://leetcode.cn/problems/path-sum/

**思路**：递归，target一直减去当前值，看叶子节点是否==剩余值；若不是叶子节点，持续递归，res==左子树和右子树递归后的或集

```
def hasPathSum(root, targetSum):
    """
    :type root: TreeNode
    :type targetSum: int
    :rtype: bool
    """
    if not root:
        return False
    
    if root.left == None and root.right ==None:
        return root.val == targetSum
    else:
        res = hasPathSum(root.left, targetSum-root.val) or hasPathSum(root.right,targetSum-root.val)
        return res
```

### 【填充每个节点的下一个右侧节点指针】

https://leetcode.cn/problems/populating-next-right-pointers-in-each-node/

**思路**：递归

节点左子树连接右子树；节点next的左子树连节点的右子树。

```
def connect(root):
    """
    :type root: Node
    :rtype: Node
    """
    
    if root == None or root.left == None:
        return root
    
    root.left.next = root.right
    
    if root.next:
        root.right.next = root.next.left
    
    connect(root.left)
    connect(root.right)
    
    return root
```


## 链表
**数组转链表**
```
def build(arr):
    h = node = ListNode(None)
    for i in range(len(head)):
        node.next = ListNode(head[i])
        node = node.next
        
    return h.next
```
### 【链表删除/定位】
该类问题很多使用dummyhead＋两个指针：slow和fast，让fast先移动后，开始让slow去定位。
##### 1. 定位倒数第n个节点
https://leetcode.cn/problems/remove-nth-node-from-end-of-list/

**思路**：让fast先跑n次，再slow和fast同步跑。
```
    dummyhead = slow = fast = ListNode(-1)
    dummyhead.next = head
    for i in range(n):
        fast = fast.next

    while fast.next:
        fast = fast.next
        slow = slow.next

    slow.next = slow.next.next
    return dummyhead.next
```

##### 2.定位中间 
https://leetcode.cn/problems/middle-of-the-linked-list

**思路**：让fast先跑两个节点，之后同时slow一次fast两次的速度跑。
```
    slow = fast = head
    if head == None or head.next == None:
        return head
    if head.next.next == None:
        return head.next
    
    fast =fast.next.next

    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
        
    if fast.next:
        return slow.next.next
    else:
        return slow.next
```

### 【链表翻转】
该类问题就是耐心将节点的后驱改成前驱。

##### 1.全部翻转
https://leetcode.cn/problems/reverse-linked-list/
```
    pre = None
    nxt = None

    while head:
        nxt =  head.next
        head.next = pre
        pre = head
        head = nxt
        
    return pre
```

##### 2.部分翻转
https://leetcode.cn/problems/reverse-linked-list-ii

**思路**：先定位第一个要改的节点，保存【前一个节点】，【第一个节点】；修改前驱后驱；【前一个节点】next连向pre（原来最后一个），【第一个节点】next连向cur（None）。
```
    nxt = None
    pre = None
    
    dummyhead = node_left_before = ListNode(None)

    dummyhead.next = head

    for i in range(left-1):
        node_left_before = node_left_before.next

    node_left = node_left_before.next
    cur = node_left_before.next

    for i in range(right-left+1):
        nxt = cur.next
        cur.next= pre
        pre = cur
        cur = nxt

    node_left_before.next = pre
    node_left.next = cur
    return dummyhead.next
```

##### 3.两两翻转
https://leetcode.cn/problems/swap-nodes-in-pairs/comments/

**思路**：递归
```
def swapPairs(head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if head == None or head.next==None:
        return head
    
    temp = head.next
    head.next = swapPairs(head.next.next)
    temp.next = head
    
    return temp
```
##### 4.每k个翻转
https://leetcode.cn/problems/reverse-nodes-in-k-group

**思路**：类似普通翻转；用循环判断满不满足k；保存每k个的头部，最后next下一个。

```
    h = head
    for i in range(k):
        if h == None:
            return head
        h = h.next
    
    pre = None
    nxt = None
    cur = head
    first = None
    for i in range(k):
        nxt = cur.next
        if i==0:
            first = cur
        cur.next = pre
        pre = cur
        cur = nxt
    first.next = reverseKGroup(cur, k)
    
    return pre
```

### 【回文判断】
https://leetcode.cn/problems/palindrome-linked-list/

**思路**：先入一半栈，弹出和原链表正着比较。
```
slow = fast = head

if head.next and head.next.next:
    fast = fast.next.next
    
while fast.next and fast.next.next:
    slow = slow.next
    fast = fast.next.next
    
slow = slow.next

stack = []
flag = True

while slow:
    stack.append(slow.val)
    slow = slow.next
    
while stack:
    if head.val != stack.pop():
        flag = False
        break
    head = head.next
    
flag
```
### 【环形链表 II】

https://leetcode.cn/problems/linked-list-cycle-ii/

**思路**：快慢指针

快速度：2

慢速度：1

z:相遇距离入口的位置

2(x+y) = x+y + n * (y+z)

x = (n-1)(z+y) + z

=> 快慢指针相遇的位置离入口的距离等于链表起始位置距离入口的距离

```
def detectCycle(head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if not head:
        return None
    fast = slow = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if slow == fast:
            break
    else:
        return None
    
    # 快慢指针相遇的位置离入口的距离等于链表起始位置距离入口的距离
    while slow!=head:
        slow = slow.next
        head = head.next
    return head
```
## 数组
### 【螺旋矩阵】
>正常螺旋

https://leetcode.cn/problems/spiral-matrix/

**思路**：需要设置两个ij的变化参数di, dj；访问过得数组直接变None，如果遇到None直接互换di, dj并且将di变负数。
```
n = len(matrix)
m = len(matrix[0])
i,j = 0, 0
di, dj = 0, 1

res = []
for _ in range(n*m):
    res.append(matrix[i][j])
    matrix[i][j] = None
    
    if matrix[(i+di)%n][(j+dj)%m] == None:
        di, dj = dj, -di
    i += di
    j += dj
```
>慢慢扩大；允许螺旋坐标不存在

https://leetcode.cn/problems/spiral-matrix-iii

**思路**：仍然需要di, dj；设置step，每两次就扩大1（用n记录）；初始要先存i, j；i, j在范围内就储存；互换di, dj并且将di变负数。
```
r, c, r0, c0 = 1, 4, 0, 0
i, j = r0, c0
di, dj = 0, 1
step = 1
n = 0

res = [[i, j]]
while len(res)< r*c:
    for k in range(step):
        i+= di
        j+= dj
        if 0<=i<r and 0<=j<c:
            res.append([i, j])
    n+=1
    if n % 2 ==0:
        step+=1
        
    di, dj = dj, -di
```

### 【删除有序数组中的重复项】

> 元素只能留下1个

**思路**：双指针,j元素必须和i不同，将j位置复制到i+1处。

```
def removeDuplicates(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    l = len(nums)
    remove = 0
    i,j = 0, 1
    while i<j and j<l:
        while i<j and j<l and nums[i] == nums[j] :
            remove += 1
            j+=1
        if j<l:
            nums[i+1] = nums[j]
        j+=1
        i+=1
    return l-remove
```

> 元素可以留2个

**思路**：双指针，j修改到i+2地方

```
def removeDuplicates(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    i, j = 0, 2
    remove = 0
    while i<j and j<len(nums):
        while j<len(nums) and nums[i] == nums[j]:
            remove += 1
            j+=1
        if j<len(nums) and i+2<len(nums):
            nums[i+2] = nums[j]
        i+=1
        j+=1

    return len(nums)-remove
```

## 字符串

### 【最长子串】
https://leetcode.cn/problems/longest-substring-without-repeating-characters/
>哈希

**思路**：用字典存最近的字符开头，记录长度；如果开头存在过，就重新开始。
```
    res, start = 0, 0
    maps = {}
    for i in range(len(s)):
        start = max(start, maps.get(s[i], -1) + 1)
        res = max(res, i - start + 1)
        maps[s[i]] = i
    return res
```

>双指针/滑动窗口

**思路**：需要一个set存/弹出没看过/看过的字符；没看过移动右指针，记录长度，看过移动左指针到不重复为止。
```
    l, r = 0, 0
    res, lookup = 0, set()
    
    while l < len(s) and r < len(s):
        if s[r] not in lookup:
            lookup.add(s[r])
            res = max(res, r-l+1)
            r += 1
        else:
            lookup.discard(s[l])
            l += 1
            
    return res
```

### 【最长公共前缀】
https://leetcode.cn/problems/longest-common-prefix

**思路**：随便选一个字符当做前缀，然后慢慢减小去匹配。

```
strs = ["flower","flow","flight"]
    
    pre = strs[0]
    for s in strs:
        while pre != s[:len(pre)]:
            if pre == '':
                return ''
            pre = pre[:-1]

    return pre
```

### 【最小覆盖子串（滑动窗口）】

https://leetcode.cn/problems/minimum-window-substring/

**思路**：滑动窗口

要素：左指针，右指针，maps记录字符可用的频率，Counter记录字符是否都用完，操作右指针，当字符频率为0，counter--，如果counter==0则表示用完，可以开始操作左指针，并且计算长度。

```
s = "ADOBECODEBANC"
t = "ABC"

def minWindow(s, t):
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    if len(s) < len(t):
        return ''
    if t=='' or s=='':
        return ''
    
    import collections
    maps = collections.Counter(t)
    maps
    l, r, head, counter, length= 0, 0, 0, len(maps), float('inf')
    while r<len(s):
        if s[r] in maps:
            maps[s[r]] -= 1
            if maps[s[r]] == 0:
                counter -=1
        r+=1
        while l<=r and counter==0:
            if s[l] in maps:
                maps[s[l]] += 1
                if maps[s[l]] > 0:
                    counter +=1
                if length > r-l:
                    length = r-l
                    head = l
            l+=1
    if length == float('inf'):
        return ''

    return s[head:length+head]
```
### 【实现实现 strStr()】
https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string

**思路**：双指针暴力；注意i只需要循环l1-l2+1次。

```
    if needle == '':
        return 0

    for i in range(len(haystack)-len(needle)+1):
        if haystack[i] == needle[0]:
            j = 1
            while j<len(needle) and haystack[i+j] == needle[j]:
                j+=1

            if j==len(needle):
                return i
    return -1
```

### 【括号生成】
https://leetcode.cn/problems/generate-parentheses

**思路**：回溯法
思路：所谓Backtracking都是这样的思路：在当前局面下，你有若干种选择。那么尝试每一种选择。如果已经发现某种选择肯定不行（因为违反了某些限定条件），就返回；如果某种选择试到最后发现是正确解，就将其加入解集

所以你思考递归题时，只要明确三点就行：选择 (Options)，限制 (Restraints)，结束条件 (Termination)。即“ORT原则”
```
res = []
def generate(s, left, right, n):
    if left == n and right == n:
        res.append(s)
    if left < n:
        generate(s+'(', left+1, right, n)
    if right < left:
        generate(s+')', left, right+1, n)
```
### 【扰乱字符串】

https://leetcode.cn/problems/scramble-string/

**思路**：递归

**递归退出条件：**

1.s1和s2字符串字母不相同：False

2.s1==s2：True

**递归主体：**

随机一个位置切割，所以用for循环；s1会被切割为s1[:i]和s1[i:]

如果不考虑交换，比较s1[:i]和s2[:i]，s1[i:]和s2[i:]是否为扰乱字符串

如果考虑交换，比较s1[i:]和s2[:-i]和s1[:i]和s2[-i:]是否为扰乱字符

**提高效率：**

存下递归重复的计算

```
def isScramble(s1, s2):
    """
    :type s1: str
    :type s2: str
    :rtype: bool
    """
    memo = {}
    
    def check(s1, s2):
        if len(s1) != len(s2) or sorted(s1)!=sorted(s2):
            return False

        if s1==s2:
            return True

        key = '%s,%s'%(s1,s2)

        if key in memo:
            return memo[key]


        for i in range(1, len(s1)):
            if check(s1[:i], s2[:i]) and check(s1[i:], s2[i:]):
                # s1被i分为s1[:i]和s1[i:];
                memo[key] = True
                return True
            if check(s1[i:], s2[:-i]) and check(s1[:i], s2[-i:]):
                # 考虑交换
                memo[key] = True
                return True

        memo[key] = False
        return False
    
    return check(s1, s2)
```
### 【复原 IP 地址】

**思路**：回溯法

backtrack(s, idx, cnt)

idx记录起始加点的位置；cnt表示有多少点了。

每个位置都加点（循环），s[:i]+'.'+s[i:]，idx=i+2，当cnt==3就结束，结果合法就加入res

```
def restoreIpAddresses(s):
    """
    :type s: str
    :rtype: List[str]
    """
    if len(s)>12 or len(s) < 4:
        return []
    
    res = []

    def check(s):

        if s.count('.')!=3:
            return False

        s = s.split('.')

        for ip in s:
            if not ip or int(ip)>255 or (len(ip)>1 and ip[0]=='0'):
                return False

        return True

    def backtrack(s, idx, cnt):
        if cnt==3:
            if check(s):
                res.append(s)
            return 

        for i in range(idx, len(s)):
            backtrack(s[:i]+'.'+s[i:], i+2, cnt+1)

    backtrack(s, 0, 0)
    return res
```
### 【不同的子序列】

https://leetcode.cn/problems/distinct-subsequences/submissions/

**思路**：动态规划

s = "babgbag", t = "bag"


dp[i][j]：s的前i个字符有多少等于t的前j个

if s[i-1]!=t[j-1]:
    
    dp[i][j] = dp[i-1][j]
    
else:
    
    1.s和t位置符号都不要 => dp[i-1][j-1]
        意思是用s的位置匹配
        
    2.s位置符号不要 => dp[i-1][j]
        意思是不用s位置的匹配

```
def numDistinct(s, t):
    """
    :type s: str
    :type t: str
    :rtype: int
    """
    
    m = len(s)
    n = len(t)
    
    if n>m:
        return 0

    dp = [[0 for y in range(n+1)] for x in range(m+1)]

    for i in range(m+1):
        dp[i][0] = 1

    # for j in range(1, n):
    #     dp[0][j] = 0

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s[i-1] != t[j-1]:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = dp[i-1][j-1] + dp[i-1][j]

    return dp[-1][-1]
```
## 动态规划

### 【最大/小动态规划】

##### 1.编辑距离

https://leetcode.cn/problems/edit-distance/

**思路**：动态规划

dp[i][j]: s1长度为i，s2长度为j时候，将s1转换s2用到的最少步数。

>si = sj => 

dp[i-1][j-1]

>si != sj =>

si删除 => dp[i-1][j]+1：

si末尾插入 => dp[i][j-1]+1

si替换 => dp[i-1][j-1] + 1
    
```
    m = len(word1)+1
    n = len(word2)+1

    dp = [[0 for y in range(n)] for x in range(m)]

    for i in range(m):
        dp[i][0] = i
    for j in range(n):
        dp[0][j] = j

    for i in range(1, m):
        for j in range(1, n):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])+1

return dp[-1][-1]
```

##### 2.最大子数组

https://leetcode.cn/problems/maximum-subarray/

**思路**：从末尾开始看，如果后面的dp是负数，那么从头计数。

if dp[i+1] <0: dp[i] = num[i]

if dp[i+1] > 0 dp[i] = dp[i+1] + nums[i]

```
dp = [0] * len(nums)
dp[len(nums)-1] = nums[len(nums)-1]

for i in range(len(nums)-2, -1, -1):
    if dp[i+1]>0:
        dp[i] = dp[i+1] + nums[i]
    else:
        dp[i] = nums[i]

return max(dp)
```

##### 3. 解码方法

https://leetcode.cn/problems/decode-ways/

**思路**：动态规划

爬楼梯

初始化第0位置为1

dp[i]

一位数或者二位数:

   dp[i] = dp[i-1] + dp[i-2]

不能一位数或二位数:
    
    si = 0: 只能二位数
    
        if withinAZ(s, i):
            dp[i+1] = dp[i-1]
        else:
            dp[i+1] = 0

    si != 0: 只能一位数

        if withinAZ(s, i):
            dp[i+1] = dp[i] + dp[i-1]
        else:
            dp[i+1] = dp[i]
            
```
def numDecodings(s):
    """
    :type s: str
    :rtype: int
    """
    if s=='' or s[0]=='0':
        return 0
    
    def withinAZ(s, i):
        return i>0 and ((s[i-1] == '2' and int(s[i])<=6) or (s[i-1]=='1'))
    
    m = len(s)
    dp = [0] * (m+1)
    dp[0] = 1
    
    for i in range(m):

        if s[i] == '0':
            if withinAZ(s, i):
                dp[i+1] = dp[i-1]
            else:
                dp[i+1] = 0
        else:
            if withinAZ(s, i):
                dp[i+1] = dp[i] + dp[i-1]
            else:
                dp[i+1] = dp[i]
            
    return dp[-1]
```
### 【范围双指针动态规划】
此类问题一般使用双指针范围的ij，还得设置一个变量去计数。
##### 1.回文子串
>回文子串的数量

https://leetcode.cn/problems/palindromic-substrings

**思路**：双指针范围的ij，还得设置一个变量去计数。
```
# 动态规划
# 判断[i，j]范围内字符串是不是回文
# if s[i] == s[j]
# i==j: 1
# j-i == 1: 1
# j-i>1: dp[i+1][j-1] => 所以要从下到上；从左到右

# if s[i] != s[j]:
# dp[i][j] = 0
s = "abcbae"
l = len(s)
m = l
n = l
dp = [[0 for y in range(n)] for x in range(m)]
cnt = 0
for i in range(m-1, -1, -1):
    for j in range(i, m):
        if s[i] == s[j]:
            if j-i <= 1:
                cnt += 1
                dp[i][j] = 1
            elif dp[i+1][j-1]:
                cnt += 1
                dp[i][j] = 1
dp,cnt
```

>定位最长的回文

https://leetcode.cn/problems/longest-palindromic-substring
**思路**：上一题基础上，设置两个索引变量，实时计算最长的距离并且保存。
```
s = "babad"
l = len(s)
dp = [[0 for y in range(l)] for x in range(l)]

indexI = indexJ = 0
for i in range(l-1, -1, -1):
    for j in range(i, l):
        if s[i] == s[j]:
            if j-i<=1:
                dp[i][j] = 1
                if j-i+1 > indexJ-indexI+1:
                    indexI = i
                    indexJ = j
            elif dp[i+1][j-1]:
                if j-i+1 > indexJ-indexI+1:
                    indexI = i
                    indexJ = j
                dp[i][j] = 1
                
s[indexI:indexJ+1]      
```

## 排序
默认从小到大排序。

##### 1.冒泡排序
**原理**：遇到小的不断交换

最佳情况：T(n) = O(n) 最差情况：T(n) = O(n2) 平均情况：T(n) = O(n2)

稳定
```
for i in range(len(x)-1):
    for j in range(0, len(x)-i -1):
        if x[j+1] < x[j]:
            x[j+1], x[j] = x[j], x[j+1]
```

##### 2.选择排序
**原理**：指定首个，选择数组中最小的和它交换。
最佳情况：T(n) = O(n2) 最差情况：T(n) = O(n2) 平均情况：T(n) = O(n2)
不稳定
```
for i in range(len(x)-1):
    index = i
    for j in range(i+1, len(x)):
        if x[j] < x[index]:
            index = j
    x[i], x[index] = x[index], x[i]
```
##### 3.插入排序
**原理**：选择元素，逐个往前比较插入到最接近且小的元素后面。循环提前将所有元素往后复制，找到之后将移动的元素粘贴。

最佳情况：T(n) = O(n) 最坏情况：T(n) = O(n2) 平均情况：T(n) = O(n2)

稳定
```
for i in range(len(x)-1):
    cur = x[i+1]
    preIndex = i
    while preIndex>=0 and x[preIndex] > cur:
        x[preIndex+1] = x[preIndex]
        preIndex -= 1
    x[preIndex+1] = cur
```

##### 4.希尔排序
**原理**：插入排序比较的间隔是相邻gap=1，希尔是最从中间比较gap=mid。

最佳情况：T(n) = O(nlog2 n) 最坏情况：T(n) = O(nlog2 n) 平均情况：T(n) =O(nlog2n)

不稳定
```
l = len(x)
gap = int(l/2)

while gap > 0:
    for i in range(gap, l):
        temp = x[i]
        preIndex = i - gap
        while preIndex >= 0 and x[preIndex] > temp:
            x[preIndex+gap] = x[preIndex]
            preIndex -= gap
        x[preIndex+gap] = temp
    gap = int(gap/2)
```

##### 5.归并排序
**原理**：分而治之的思想。从数组中间分割，递归将子集先排序然后在新的数组里合并。强调合并的过程。

最佳情况：T(n) = O(n) 最差情况：T(n) = O(nlogn) 平均情况：T(n) = O(nlogn)

稳定
```
def MergeSort(array):
    if len(array) < 2:
        return array
    l = len(array)
    mid = int(l/2)
    left = array[0: mid]
    right = array[mid: l]
    return Merge(MergeSort(left), MergeSort(right))

def Merge(left, right):
    res = [0] * (len(right) + len(left))
    i, j = 0, 0
    for index in range(len(res)):
        if i >= len(left):
            res[index] = right[j]
            j += 1
        elif j >= len(right):
            res[index] = left[i]
            i += 1
        elif left[i] > right[j]:
            res[index] = right[j]
            j += 1
        else:
            res[index] = left[i]
            i += 1
    return res
```

##### 6.快速排序
**原理**：随机选取比较元素，小的放左边边，大的放右边，以此类推。
过程：随机选取元素，移动末尾，双指针i，j，比较元素：大的插入j+1后，并且j++，小的和i+1的元素对调，并且i++和j++。结束后末尾元素放到i+1处，返回当前索引p。合并只需要递归p左右的子集。

最佳情况：T(n) = O(nlogn) 最差情况：T(n) = O(n2) 平均情况：T(n) = O(nlogn)

不稳定
```
def Partition(array, start, end):
    pivot = int(start + random.random() * (end - start + 1))
    array[pivot], array[end] = array[end], array[pivot]
    i = j = start - 1
    for index in range(start, end):
        if array[end] > array[index]:
            array[index], array[i+1] = array[i+1], array[index]
            i += 1
            j += 1
        elif array[end] < array[index]:
            j+1
    array[i+1], array[end] = array[end], array[i+1]
    return i+1

def QuickSort(array, start, end):
    if len(array) < 1 or start < 0 or end >= len(array) or start> end:
        return None
    p = Partition(array, start, end)
    if p > start:
        QuickSort(array, start, p - 1)
    if p < end:
        QuickSort(array, p+1, end)
    return array
```

##### 7.堆排序
**原理**：建立上大下小的二叉树（大顶堆）。首位互换，大的放到末尾，重新建堆。

最佳情况：T(n) = O(nlogn) 最差情况：T(n) = O(nlogn) 平均情况：T(n) = O(nlogn)

不稳定
```
def heapify(array, n, i):
    largest = i
    lson = i * 2 + 1
    rson = i * 2 + 2
    
    if lson < n and array[largest] < array[lson]:
        largest = lson
    if rson < n and array[largest] < array[rson]:
        largest = rson
    if largest != i:
        array[largest], array[i] = array[i], array[largest]
        heapify(array, n, largest)
    
n = len(x)
# 建堆
for i in range(int(n/2-1),-1,-1): # 下标为i的节点父节点是(i-1) / 2 整除; (i-1-1)/2
    heapify(x, n, i)
for i in range(n-1, -1, -1):
    x[i], x[0] = x[0], x[i]
    heapify(x, i, 0)
```

##### 8.计数排序
**原理**：空间换时间。计数对应元素；新建累加数组，累加前一个元素；新建结果数组，根据原数组和累加数组计算索引。

最佳情况：T(n) = O(n+k) 最差情况：T(n) = O(n+k) 平均情况：T(n) = O(n+k) T（n）=O（n+k）T（N）=0（n=0）T（n）=O（N+k）[输入0-k之间整数]
```
# find max
x = d.copy()
max_value = -1
for data in x:
    if data>max_value:
        max_value = data
        
count = [0] * (max_value+1)

# counting
for i in range(len(x)):
    count[x[i]] += 1

accumulate = [0] * len(count)
accumulate[0] = count[0]
for i in range(1, len(accumulate)):
    accumulate[i] = count[i]+accumulate[i-1]

res = [0] * len(x)
for number in x:
    res[accumulate[number]-1] = number
    accumulate[number] -=1
```
# 二分查找

### 【基本方法】
一直往中间查找。
```
arr = [1,4,6,10,18,20]
target = 18

left = 0
right = len(arr)-1
while left <= right:
    mid = int(left + (right - left)/2)

    if target > arr[mid]:
        left = mid + 1
    elif target < arr[mid]:
        right = mid - 1
        mid = int((left+right)/2)
    elif target == arr[mid]:
        print(mid)
        break
```

### 【实现开根号】

https://leetcode.cn/problems/sqrtx/

**思路**：二分搜索

在0和 2x之间一个一个试，但用mid

```
def mySqrt(x):
    """
    :type x: int
    :rtype: int
    """
    l,r = 0, 2*x
    while l<=r:  

        mid = int((l+r)/2)
        # mid = l + (r-l)>>1

        if mid*mid<=x and (mid+1)*(mid+1)>x:
            return mid
        
        if mid*mid<x:
            l = mid + 1
            
        else:
            r = mid - 1

```

### 【实现Pow(x,n)】

https://leetcode.cn/problems/powx-n/

**思路**：递归；位运算/2

```
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n==0:
            return 1
        if n<0:
            return 1 / self.myPow(x, -n)
        if n&1:
            return x * self.myPow(x*x, n>>1)
        else:
            return self.myPow(x*x, n>>1)
```

### 【搜索旋转排序数组】

>数组元素各不相同

https://leetcode.cn/problems/search-in-rotated-sorted-array/

**思路**：在二分查找基础上，分类讨论mid，target和r/l的关系

以右边作为基准，如果mid小于等于右边（mid-r未分割），比较target，mid，r的大小。

如果mid大于右边（l-mid未分割），先看没有被分割的情况：比较target，mid，l的大小。

```
nums = [4,5,6,7,0,1,2]
target = 0

def search(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """

    l, r = 0, len(nums) - 1

    while l<=r:
        mid = int((l+r)/2)

        if nums[mid] == target:
            return mid

        elif nums[mid] <= nums[r]:
            if nums[mid] < target <= nums[r]:
                l = mid+1
            else:
                r = mid-1
        else:
            if nums[l] <= target < nums[mid]:
                r = mid-1
            else:
                l = mid+1

    return -1
```

>数组元素允许相同

https://leetcode.cn/problems/search-in-rotated-sorted-array-ii/

**思路**：思路和上面一样，但是注意元素允许相同，所以要多一类考虑。

以右边作为基准，如果mid小于右边（mid-r未分割），比较target，mid，r的大小。

如果mid大于右边（l-mid未分割），先看没有被分割的情况：比较target，mid，l的大小。

如果mid等于右边，那么递归调用两种情况（l+1或者r-1）。

```
def search(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: bool
    """

    def searchloop(nums,target,l,r):

        while l<=r:

            mid = l+((r-l)>>1)
            if nums[mid] == target:
                return True

            elif nums[mid]< nums[r]:
                if nums[mid]<target<=nums[r]:
                    l = mid+1
                else:
                    r = mid-1
            elif nums[mid]> nums[r]:
                if nums[l] <= target < nums[mid]:
                    r = mid-1
                else:
                    l = mid+1
            else:
                res1 = searchloop(nums, target, mid+1, r)
                res2 = searchloop(nums, target, l, mid-1)

                return res1 or res2

        return False
    return searchloop(nums,target,0, len(nums)-1) 

```
# 贪心算法

### 【跳跃游戏】

https://leetcode.cn/problems/jump-game-ii/

计算最大范围，如果到了最底部则步骤+1返回；如果当前位置到了上一次的最大位置，则步骤+1

```
def jump(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    cur_end, cur_furthest, step = 0, 0, 0

    for i in range(len(nums)-1):

        cur_furthest = max(nums[i]+i, cur_furthest)

        if cur_furthest >= len(nums)-1:
            step += 1
            return step

        if cur_end == i:
            cur_end = cur_furthest
            step+=1

    return step
```

# 单调栈

### 【柱状图中最大的矩形】

https://leetcode.cn/problems/largest-rectangle-in-histogram/

**思路**：单调栈

现在height首位插入0方便统一运算，栈先加入索引0，保持栈底到栈顶对应高度从小到大，如果遇到h[i]小于栈顶元素，栈顶弹出栈作为高，宽=i-stack[-1] - 1（当前位置-弹出元素前一个位置-1），求面积，并持续弹出运算。

```
heights = [2,1,5,6,2,3]
def largestRectangleArea(heights):
    """
    :type heights: List[int]
    :rtype: int
    """
    heights.insert(0,0)
    heights.append(0)
    stack = [0]
    res = 0
    for i in range(1,len(heights)):
        while stack and heights[stack[-1]]>=heights[i]:
            mid = stack.pop()
            if stack:
                area = (i-stack[-1] -1) * heights[mid]
                res = max(area, res)
        stack.append(i)
    return res
```


### 【接雨水】

https://leetcode.cn/problems/trapping-rain-water

**思路**：单调栈

保持栈底到栈顶从大到小，如果h[i]比栈顶大了，说明可以接雨水了，弹出栈顶idx

w = i - stack[-1] -1

h = min(height[stack[-1]] - height[idx], height[i]-height[idx])

计算面积，累积雨水

```
def trap(height):
    """
    :type height: List[int]
    :rtype: int
    """
    stack = []
    res = 0
    for i in range(len(height)):
        while stack and height[i] > height[stack[-1]]:
            idx = stack.pop()

            if stack:
                w = i - stack[-1] -1
                h = min(height[stack[-1]] - height[idx], height[i]-height[idx])
                area = w*h
            else:
                area = 0

            res += area  
        stack.append(i)

    return res
```
# 其他


