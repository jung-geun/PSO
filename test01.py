# 반복문을 사용해서 자동 생성하는 python 코드


def pibonachi(n):
    if n <= 1:
        return n
    else:
        return pibonachi(n - 1) + pibonachi(n - 2)


print(pibonachi(10))
