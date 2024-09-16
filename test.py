
class Polyn:

    def get_degree(self):
        for i in range(self.N,-1,-1):
            if self.coeffs[i] != 0:
                return i
        return 0
    def __init__(self,n : int) -> None:
        self.coeffs = [0 for i in range(n)]
        self.N = len(self.coeffs) - 1
        self.degree = self.get_degree()


def get_n_coeffs(N):
    return [0 for i in range(N)]

def polymodq(a,q):
    return [i%q for i in a]



#Compute the product of two polynomials given by a(x),b(x)
# in the quotient ring R = (Z[x]/(x^N - 1))
def poly_prod(a,b,N):
    #initialize output array
    c = [0 for i in range(N)]
    # Follow algorithm in page 414 of textbook
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if (i+j)%N == k:    
                    c[k] += (a[i]*b[j])
    return c 

#calculate a = b*k + r
def poly_divide(a,b):
    k = get_n_coeffs(len(a))





def extended_gcd(a, b):
    x0, x1, y0, y1 = 1, 0, 0, 1
    while b != 0:
        q, a, b = a // b, b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


# #Find inverse of polynomial given by a(x) in (Z/qZ)[x]/(x^N - 1)
# def extended_gcd_poly(a,N,q):


print(poly_prod([1,-2,0,4,-1],[3,4,-2,5,2],5))


def factorOut2(n):
    if n%2 != 0:
        raise ValueError()
    k = 0
    q=n
    while q%2 == 0:
        k+=1
        q//=2
    return k,int(q)

smallPrimes = [2,3,5,7,11,13,17]

def isDiv(n):
    for p in smallPrimes:
        if n%p == 0:
            return True
        return False

def isPrimeWit(n,a):
    # Your code here
    # Return True or False
    if n==2:
        return True
    if isDiv(n) or 1 < math.gcd(n,a) < n:
        return False
    k,q = factorOut2(n-1)

    a = pow(a,q,n)
    if a==1:
        return True
    neg_one = (-1)%n
    for _ in range(k):
        if a==neg_one:
            return True
        a = pow(a,2,n)
    return False

def isPrime(n):
    itrs = 10
    for _ in range(itrs):
        wit = random.randint(2,n-1)
        num_is_prime = isPrimeWit(n,wit)
        if not num_is_prime:
            return False
    return True

def nBitPrime(nbits):
    lb = int(2**(nbits-1))
    ub = int(2**nbits)
    while True:
        a =random.randint(lb,ub)
        if isPrime(a):
            return a
    raise ValueError()

def getCandidates(nbits,q):
    lb = int(2**(nbits-1))
    ub = int(2**nbits)
    st = lb
    for i in range(lb,ub):
        if i%q == 1:
            st =i
            break
    print("Found init")
    ls = []
    while st < ub:
        ls.append(st)
        st+=q
    return ls


def makeQP(qbits,pbits):


    itrs = 10
    while True:
        print("Trying new q")
        q = nBitPrime(qbits)
        lb = (2**(pbits-1))
        ub = (2**pbits)-1
        lbk = (lb//q) + 1
        ubk = ub//q
        for _ in range(itrs):
            k = random.randint(lbk,ubk)
            p = (q*k) + 1
            if isPrime(p):
                return q,p


def extended_gcd(a, b):
    if b.degree() > a.degree():
        a,b = b,a 

    x0, x1, y0, y1 = 1, 0, 0, 1
    while b != 0:
        quo,rem = a.pseudo_quo_rem(b)
        q, a, b = quo, b, rem
        x0, x1 = x1, x0 - (q * x1)
        y0, y1 = y1, y0 - (q * y1)
    return a, x0, y0