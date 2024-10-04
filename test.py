n, q = map(int, input().split())
a = list(map(int, input().split()))
for i in range(q):
    str = input()
    if str[0] == '+':
        l, r, x = map(int, str[2:].split())
        for i in range(l-1, r):
            a[i] += x
    else:
        l, r, k, b = map(int, str[2:].split())
        ans = 0
        for i in range(l-1, r):
            z = min(a[i], k * (i+1) + b)
            if z > ans: ans =z
        print(ans)