% ul1.m

function beta = ul1(z, Lam, lambda)

beta = sign(z)*max(0, abs(z) - Lam*lambda);
