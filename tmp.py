import sage.all as sage
q=2048


PolyRing = sage.PolynomialRing(sage.ZZ, 'x')



qRing = sage.PolynomialRing(sage.IntegerModRing(q),"x")

x = qRing([1,1,-1])

y = qRing([-1,0,0,0,1])

Rq = qRing.quotient(y)

a = Rq(x.list())

