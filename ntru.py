# Import SageMath
import sage.all as sage
import random 
import math
import time

### An implementation of the NTRUEncrypt cryptosystem.


# q is always a power of 2
QBASE  = 2

# Get a list of the element e repeated n times
def get_n_list(e,n):
    return [e for _ in range(n)]

#Choose the public parameters 
def choose_public_parameters(N=None):
    if N is None:
        N = 128
    p = 3
    q = 2048 #p,q are the recommended values for any value of N
    d = N//3 #N/3 is a recommended value for d
    return (N,p,q,d)

#Get the coefficients of the ternary polynomial T(d1,d2) unshuffled
def get_ternary_coeffs(N,d1,d2):
    return get_n_list(1,d1) + get_n_list(-1,d2) + get_n_list(0,N-(d1+d2))


def inverse_prime_power(flist,modpol:str,p,e):
    """
        Return the inverse of a polynomial mod p^e. Adapted from the IEEE specification 1363.1-2008 
    Args:
        flist: The coefficients of the polynomial f(x) whose inverse is required
        modpol (str): The monic polynomial that gives the convolution polynomial ring
        p: A prime number
        e : The power to raise e to

    Returns:
        G(x) s.t f(x) G(x) = 1 mod (p^e) in (Z[x]/modpol)
    """
    Rcurp = sage.PolynomialRing(sage.IntegerModRing(p),"x").quotient(modpol)
    try:
        curinv = Rcurp(flist).inverse() 
    except:
        return None
    n = 2
    while e > 0:
        Rcurp = sage.PolynomialRing(sage.IntegerModRing(p**n),"x").quotient(modpol)
        fcur = Rcurp(flist)
        curinv = Rcurp(curinv.list())
        curinv = curinv*(2-(curinv*fcur))
        n = 2*n
        e = math.floor(e/2)
    return curinv
        
#Generate a private,public key using the public parameters (N,p,q,d)
def key_creation(N,p,q,d):
    #The modulus monic polynomial
    monic_pol = f"x^{N} - 1"
    # q = 2^qpow
    qpow = math.log(q,2)

    #The initial coeffcients
    f_init_coeffs = get_ternary_coeffs(N,d+1,d)

    #The ring Z[x]/(x^N - 1)
    R = sage.PolynomialRing(sage.ZZ, 'x').quotient(monic_pol)

    #The ring (Z/qZ)[z]
    qRing = sage.PolynomialRing(sage.IntegerModRing(q),"x")
     #The ring (Z/pZ)[z]
    pRing = sage.PolynomialRing(sage.IntegerModRing(p),"x")
    pRingpol = (pRing.gen()**N) - 1
    Rq = qRing.quotient(monic_pol)
    Rp = pRing.quotient(monic_pol)

    #Keep generating f(x) in T(d+1,d) until we find one that is invertible in Rp and Rq
    while True:
        random.shuffle(f_init_coeffs)
        
        f = R(f_init_coeffs)

        #Find inverse in Rq
        try:
            Fq_coeffs = inverse_prime_power(f_init_coeffs,monic_pol,QBASE,qpow).list()
            Fq = Rq(Fq_coeffs)
        except:
            print("no q inverse")
            continue

        #Find inverse in Rp
        try:
            Fp_coeffs = pRing(f_init_coeffs).inverse_mod(pRingpol).list()
            Fp = Rp(Fp_coeffs)
        except:
            print("no p inverse")
            continue
        # Create a random g(x) in T(d,d)
        g_coeffs =  get_ternary_coeffs(N,d,d)
        random.shuffle(g_coeffs)
        gq = Rq(g_coeffs)

        #Create public key
        h = Fq*gq

        #Return (f,Fp,h) where (f,Fp) is the private key and h is the public key
        return (f.list(),Fp,h)

#Encrypt a message given the public parameters (N,p,q,d) and the coefficents of the public key h and the coefficients of the plaintext m
def encryption(N,p,q,d,h_coeffs,m_coeffs):
    #Generate a random element r(x) in T(d,d)
    r_coeffs= get_ternary_coeffs(N,d,d)
    random.shuffle(r_coeffs)

    
    monic_pol = f"x^{N} - 1"
    qRing = sage.PolynomialRing(sage.IntegerModRing(q),"x")
    Rq = qRing.quotient(monic_pol)

    mq = Rq(m_coeffs)
    rq = Rq(r_coeffs)
    hq = Rq(h_coeffs)

    e = (p*hq*rq) + mq 
    return e.list()


#Centre lift a the polynomial given by coeffs from Rq to R
def centre_lift(coeffs,q):
    nlist = [int(i)%q for i in coeffs]
    qdiv2 = q//2
    for idx in range(len(nlist)):
        if nlist[idx] <= qdiv2:
            continue
        nlist[idx] = (nlist[idx]-q)   
    return nlist

#Decrypt a ciphertext e using public parameters (N,p,q,d) and private keys (f,Fp)
def decryption(N,p,q,d,f,Fp,e):
    
    monic_pol = f"x^{N} - 1"
    qRing = sage.PolynomialRing(sage.IntegerModRing(q),"x")
    pRing = sage.PolynomialRing(sage.IntegerModRing(p),"x")
    Rq = qRing.quotient(monic_pol)
    Rp = pRing.quotient(monic_pol)

    eq = Rq(e)
    f = Rq(f)
    a = f*eq
    a_centre_lift_coeff = centre_lift(a.list(),q)
    b = Fp * Rp(a_centre_lift_coeff)
    return b


#Check that decryption recovered the correct plaintext
def check_correct_decryption(blist,m_coeffs):
    if blist != m_coeffs:
        raise ValueError("Decryption failed")
    

#Plot figure used in paper. REQUIRES extra non standard libraries
def plot_figure():
    #NOTE: Requires matplot lib
    import matplotlib.pyplot as plt
    import numpy as np
    N_values = [7,37,151,227,359,439,571,631,743]
    time_taken = []
    ITRS = 500
    for N_val in N_values:
        print(f"Running for N={N_val}",flush=True)
        N,p,q,d = choose_public_parameters(N_val)
        f,Fp, h = key_creation(N,p,q,d)
        start_time = time.perf_counter_ns()
        for _ in range(ITRS):
            m_coeffs = [random.randint(math.ceil(-p/2),p//2) for _ in range(N)]
            e = encryption(N,p,q,d,h.list(),m_coeffs)
            b  = decryption(N,p,q,d,f,Fp,e)
            check_correct_decryption(b.list(),m_coeffs)
        end_time = time.perf_counter_ns()
        total_time = (end_time-start_time)/(10**6) #convert to milliseconds from nanoseconds
        time_taken.append(total_time)
    plt.figure(figsize=(10, 6))

    # Plotting the data points

    X_vals  = np.array(N_values)
    Y_vals = time_taken #np.log(np.array(time_taken))
    plt.scatter(X_vals, Y_vals, color='blue', label='Data points (N_val, time_taken)')

    # Labeling each point with its N value
    for i, txt in enumerate(X_vals):
        plt.annotate(f"N={txt}", (X_vals[i], Y_vals[i]))
    print("Times are ",time_taken)

    x_range = np.linspace(min(N_values), max(N_values), 400)
    coeffs = np.polyfit(X_vals,Y_vals,deg=2)
    p = np.poly1d(coeffs)
    plt.plot(x_range,p(x_range),color="red",label="Quadratic line of best fit")

    # Adding axis labels
    plt.xlabel('N_val')
    plt.ylabel('Time taken (milliseconds)')
    print("coeffs are ",coeffs)
    # Adding legend with additional information
    plt.legend(title=f"Parameters: p = 3, q = 2048, d = N/3")
    
    # Show the plot
    plt.show()


# plot_figure()
### Test code . Use Miller-Rabin test to generate random prime values for public parameter N
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
    itrs = 100
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

#RUN THIS FUNCTION to test ntru end to end
def test_ntru():
    print("Choosing random prime value of N")
    N_val  = nBitPrime(random.randint(9,10)) #Choose a random 9 or 10 bit prime which is the range of recommended values for secure systems
    N,p,q,d = choose_public_parameters(N_val) # Choose public parameters based on selected value of N

    print(f"Public parameters are N={N},p={p},q={q},d={d}")
    
    KEYGEN = 5 # Number of keys to generate
    ITRS = 20 # Number of encyrption/decryptions to run per key pair 
    
    for _ in range(KEYGEN):
        print("Generating new private,public keys")
        f,Fp, h = key_creation(N,p,q,d)
        print("Generated new private,public keys")
        start_time = time.perf_counter_ns()
        for _ in range(ITRS):
            #Generate a random plaintext message m in Rp
            m_coeffs = [random.randint(0,p-1) for _ in range(N)]
            #Ciphertext
            e = encryption(N,p,q,d,h.list(),m_coeffs)
            #Decrypted plaintext
            b  = decryption(N,p,q,d,f,Fp,e)
            #Check that decrypted message is equal to original plaintext
            check_correct_decryption(b.list(),m_coeffs)
        end_time = time.perf_counter_ns()
        total_time = (end_time-start_time)/(10**6) #convert to milliseconds from nanoseconds
        print(f"Succesfully encrypted/decrypted {ITRS} messages in {total_time} milliseconds")

if __name__ == "__main__":
    test_ntru()