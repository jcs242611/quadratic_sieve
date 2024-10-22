from gmpy2 import mpz, is_prime, powmod, mod, isqrt, mul, gcd, log, exp, remove, f_div
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def solveCongruence(smooths, xList, dependencies, n, t):
    X = mpz(1)
    Y = mpz(1)

    for row in dependencies:
        X = mul(X, xList[row])
        Y = mul(Y, smooths[row])

    if Y < 0:
        Y = mod(Y, n)

    X = mod(X, n)
    Y = mod(isqrt(Y), n)

    factor = gcd(X - Y, n)
    if factor == 1 or factor == n:
        print(
            f"Trail {t} -> Trivial factor found. Try again with another dependency.")
        return None
    else:
        print(f"Trail {t} -> Non-trivial factor found: ", factor)
        return factor


def performingGaussianElimination(matrix, rowCount, colCount, smooths, xList, n):
    print("Peforming Gaussian Elimination...")
    marked = [False] * rowCount

    for pivotCol in range(colCount):
        pivotRow = -1
        for row in range(rowCount):
            if matrix[row][pivotCol] == 1:
                pivotRow = row
                marked[pivotRow] = True
                break

        if pivotRow == -1:
            continue

        for col in range(colCount):
            if col != pivotCol and matrix[pivotRow][col] == 1:
                for row in range(rowCount):
                    matrix[row][col] = (
                        matrix[row][pivotCol] + matrix[row][col]) % 2

    nullRows = []
    for i in range(rowCount):
        if marked[i] == False:
            nullRows.append([matrix[i], i])

    trail = 0
    for nullRow in nullRows:
        trail += 1
        dependency = [nullRow[1]]
        setColumns = []
        for i in range(colCount):
            if nullRow[0][i] == 1:
                setColumns.append(i)
        for row in range(rowCount):
            if marked[row]:
                for setColumn in setColumns:
                    if matrix[row][setColumn] == 1:
                        dependency.append(row)
                        break
        factor = solveCongruence(smooths, xList, dependency, n, trail)
        if factor:
            print(f"Factors of {n}: {factor} and {n // factor}.")
            return

    print("No non-trivial factors found.")
    return


def getExponents(Q, factorBase):
    exp = []

    if Q < 0:
        exp.append(mpz(1))
        Q = -Q
    else:
        exp.append(mpz(0))

    for prime in factorBase[1:]:
        Q, powerCount = remove(Q, prime)
        exp.append(mod(powerCount, 2))

    return exp if Q == 1 else None


def modularSqrt(n, p):
    if n == 0:
        return 0

    if powmod(n, f_div(p - 1, 2), p) != 1:
        return None

    if p % 4 == 3:
        temp = powmod(n, f_div(p + 1, 4), p)
        return [temp, p - temp]

    q, s = remove(p-1, 2)

    z = 2
    while powmod(z, f_div(p - 1, 2), p) != p - 1:
        z += 1

    m = s
    c = powmod(z, q, p)
    t = powmod(n, q, p)
    r = powmod(n, f_div(q + 1, 2), p)

    while t != 0 and t != 1:
        t2i = t
        i = 0
        for i in range(1, m):
            t2i = powmod(t2i, 2, p)
            if t2i == 1:
                break

        b = powmod(c, 2 ** (m - i - 1), p)
        m = i
        c = powmod(b, 2, p)
        t = mod(mul(t, c), p)
        r = mod(mul(r, b), p)

    return [r, p - r]


def buildMatrix(smooths, factorBase):
    print("Building Matrix...")

    matrix = []
    for smooth in smooths:
        matrix.append(getExponents(smooth, factorBase))

    return matrix


def getSmooths(factorBase, n, bound):
    print(f"Process {rank}: Finding Smooths...")

    localSmooths = []
    localXList = []
    localBound = f_div(bound, size)

    m = isqrt(n)
    s = mul(rank, localBound) + m
    e = s + localBound

    Q = [mpz((x)**2 - n) for x in range(s, e)]
    QCopy = Q.copy()

    for prime in factorBase[1:]:
        qRs = modularSqrt(n, prime)
        for qR in qRs:
            sP = mod(qR-s, prime)
            for p in range(sP, localBound, prime):
                QCopy[p] = remove(QCopy[p], prime)[0]

    for i, q in enumerate(QCopy):
        if q == 1:
            # print(f"Process {rank}: Q({i+s}) is pt-smooth")
            localSmooths.append(Q[i])
            localXList.append(mpz(i+s))

    print(f"Process {rank}: Sieving done!")
    print(f"Process {rank}: Found {
          len(localSmooths)} smooths in sieve interval [{s}, {e})")

    all_smooths = comm.gather(localSmooths, root=0)
    all_xList = comm.gather(localXList, root=0)

    if rank == 0:
        smooths = [smooth for sublist in all_smooths for smooth in sublist]
        xList = [x for sublist in all_xList for x in sublist]

        print(f"Total smooths found: {len(smooths)}")

        return smooths, xList
    else:
        return [], []


def isQuadraticResidue(a, p):
    return powmod(a, (p-1) // 2, p) == 1


def getFactorBase(n, B):
    print("Finding Factor Base...")
    factorBase = [mpz(-1)]
    # print("-1 is added to the factor base")

    for i in range(2, B+1):
        if is_prime(i) and isQuadraticResidue(n, i):
            factorBase.append(mpz(i))
            # print("{i} is added to the factor base")

    return factorBase


def QSF(n, b, i):
    if rank == 0:
        factorBase = getFactorBase(n, b)
    else:
        factorBase = []

    factorBase = comm.bcast(factorBase, root=0)
    smooths, xList = getSmooths(factorBase, n, i)

    if rank == 0:
        matrix = buildMatrix(smooths, factorBase)
        performingGaussianElimination(matrix, len(
            smooths), len(factorBase), smooths, xList, n)


def getBase(n, c=1.0, alpha=0.5):
    lnN = log(n)
    lnlnN = log(lnN)
    return mpz(exp(c * (lnN**alpha) * (lnlnN**(1 - alpha))))


def main():
    if (rank == 0):
        n = input("Enter the number you want to factorize: ")
        if not n.isnumeric():
            print("Enter a valid number!!")
            MPI.Comm.Abort(MPI.COMM_WORLD)
        n = mpz(n)
    else:
        n = mpz(0)

    n = comm.bcast(n, root=0)
    # m = isqrt(n)
    # B = getBase(n)
    QSF(n, mpz(10000), mpz(5000000))


if __name__ == "__main__":
    main()
