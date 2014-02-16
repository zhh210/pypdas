'''
Python implementation of MINRES, a translation of Jeffery Kline's implementation to the OO manner.
Original author's code available on http://pages.cs.wisc.edu/~kline/cvxopt/
'''

from cvxopt import matrix
from cvxopt.lapack import *
from cvxopt.blas import *
from math import sqrt
from copy import copy

# 2009_05_11 - removed unnecessary declaration of '_y' and '_w'
# 2009_06_07 - synchronized with May 11 version hosted at Stanford
# 2009_10_26 - corrected minor error in SymOrtho,  thanks to Mridul Aanjaneya at Stanford
# 2014_02_14 - changed interface to the object-oriented way by Zheng
"""
a,b are scalars

On exit, returns scalars c,s,r
"""
def SymOrtho(a,b):
    aa=abs(a)
    ab=abs(b)
    if b==0.:
        s=0.
        r=aa
        if aa==0.:
            c=1.
        else:
            c=a/aa
    elif a==0.:
        c=0.
        s=b/ab
        r=ab
    elif ab>=aa:
        sb=1
        if b<0: sb=-1
        tau=a/b
        s=sb*(1+tau**2)**-0.5
        c=s*tau
        r=b/s
    elif aa>ab:
        sa=1
        if a<0: sa=-1
        tau=b/a
        c=sa*(1+tau**2)**-0.5
        s=c*tau
        r=a/c
        
    return c,s,r

"""
function [ x, istop, itn, rnorm, Arnorm, Anorm, Acond, ynorm ] = ...
           minres( A, b, M, shift, show, check, itnlim, rtol )

%        [ x, istop, itn, rnorm, Arnorm, Anorm, Acond, ynorm ] = ...
%          minres( A, b, M, shift, show, check, itnlim, rtol )
%
% minres solves the n x n system of linear equations Ax = b
% or the n x n least squares problem           min ||Ax - b||_2^2,
% where A is a symmetric matrix (possibly indefinite or singular)
% and b is a given vector.  The dimension n is defined by length(b).
%
% INPUT:
%
% "A" may be a dense or sparse matrix (preferably sparse!)
% or a function handle such that y = A(x) returns the product
% y = A*x for any given n-vector x.
%
% If M = [], preconditioning is not used.  Otherwise,
% "M" defines a positive-definite preconditioner M = C*C'.
% "M" may be a dense or sparse matrix (preferably sparse!)
% or a function handle such that y = M(x) solves the system
% My = x given any n-vector x.
%
% If shift ~= 0, minres really solves (A - shift*I)x = b
% (or the corresponding least-squares problem if shift is an
% eigenvalue of A).
%
% When M = C*C' exists, minres implicitly solves the system
%
%            P(A - shift*I)P'xbar = Pb,
%    i.e.               Abar xbar = bbar,
%    where                      P = inv(C),
%                            Abar = P(A - shift*I)P',
%                            bbar = Pb,
%
% and returns the solution      x = P'xbar.
% The associated residual is rbar = bbar - Abar xbar
%                                 = P(b - (A - shift*I)x)
%                                 = Pr.
%
% OUTPUT:
%
% x      is the final estimate of the required solution
%        after k iterations, where k is return in itn.
% istop  is a value from [-1:9] to indicate the reason for termination.
%        The reason is summarized in msg[istop+2] below.
% itn    gives the final value of k (the iteration number).
% rnorm  estimates norm(r_k)  or norm(rbar_k) if M exists.
% Arnorm estimates norm(Ar_{k-1}) or norm(Abar rbar_{k-1}) if M exists.
%        NOTE THAT Arnorm LAGS AN ITERATION BEHIND rnorm.

% Code authors:Michael Saunders, SOL, Stanford University
%              Sou Cheng Choi,  SCCM, Stanford University
%
% 02 Sep 2003: Date of Fortran 77 version, based on 
%              C. C. Paige and M. A. Saunders (1975),
%              Solution of sparse indefinite systems of linear equations,
%              SIAM J. Numer. Anal. 12(4), pp. 617-629.
%
% 02 Sep 2003: ||Ar|| now estimated as Arnorm.
% 17 Oct 2003: f77 version converted to MATLAB.
% 03 Apr 2005: A must be a matrix or a function handle.
% 10 May 2009: Parameter list shortened.
%              Documentation updated following suggestions from
%              Jeffery Kline <jeffery.kline@gmail.com>
%              (author of new Python versions of minres, symmlq, lsqr).

% % Known bugs: % 1. ynorm is currently mimicking ynorm in symmlq.m.
% It should be sqrt(x'Mx), but doesn't seem to be correct.  % Users
really want xnorm = norm(x) anyway.  It would be safer % to compute it
directly.  % 2. As Jeff Kline pointed out, Arnorm = ||A r_{k-1}|| lags
behind % rnorm = ||r_k||.  On singular systems, this means that a good
% least-squares solution exists before Arnorm is small enough % to
recognize it.  The solution x_{k-1} gets updated to x_k % (possibly a
very large solution) before Arnorm shuts things % down the next
iteration.  It would be better to keep x_{k-1}.
%------------------------------------------------------------------
"""

class MINRES(object):
    'MINRES class to hold minres iterations.'
    def __init__(self,A,b,x0=None):
        self.option = dict()
        self.option['M'] = None
        self.option['shift'] = 0.0
        self.option['show'] = False
        self.option['check'] = False
        self.option['itnlim'] = None   
        self.option['rtol'] = 1e-7
        self.option['eps'] = 2.2e-16
        self.option['msg'] = (' beta2 = 0.  If M = I, b and x are eigenvectors '   ,
           ' beta1 = 0.  The exact solution is  x = 0       '   ,
           ' A solution to Ax = b was found, given rtol     '   ,
           ' A least-squares solution was found, given rtol '   ,
           ' Reasonable accuracy achieved, given eps        '   ,
           ' x has converged to an eigenvector              '   ,
           ' acond has exceeded 0.1/eps                     '   ,
           ' The iteration limit was reached                '   ,
           ' A  does not define a symmetric matrix          '   ,
           ' M  does not define a symmetric matrix          '   ,
           ' M  does not define a pos-def preconditioner    ')

        self.A = A
        self.b = b
        self.n = A.size[0]
        self.iter = 0
        if x0 is not None:
            self.x = x0
        else:
            self.x = matrix(0., (self.n,1))

        #--------

        n = self.n
        if self.option['itnlim'] is None: self.option['itnlim'] = 5*n

        self.precon=True
        if self.option['M'] is None:        self.precon=False
        
        if self.option['show']:
            print '\n minres.m   SOL, Stanford University   Version of 10 May 2009'
            print '\n Solution of symmetric Ax = b or (A-shift*I)x = b'
            print '\n\n n      =%8g    shift =%22.14e' % (n,shift)
            print '\n itnlim =%8g    rtol  =%10.2e\n'  % (itnlim,rtol)


        self.istop = 0;   self.itn   = 0;   self.Anorm = 0;    
        self.Acond = 0;   self.rnorm = 0;   self.ynorm = 0;   
        self.done  = False;


        """
        %------------------------------------------------------------------
        % Set up y and v for the first Lanczos vector v1.
        % y  =  beta1 P' v1,  where  P = C**(-1).
        % v is really P' v1.
        %------------------------------------------------------------------
        """
        self.y     = +b;
        self.r1    = +b;
        if self.precon:        M(self.y)              # y = minresxxxM( M,b ); end
        self.beta1 = dotu(b, self.y)                  # beta1 = b'*y;

        """
        %  Test for an indefinite preconditioner.
        %  If b = 0 exactly, stop with x = 0.
        """
        if self.beta1< 0: self.istop = 8;  show = True;  self.var['done'] = True;
        if self.beta1==0: show = True;  self.var['done'] = True;


        if self.beta1> 0: self.beta1  = sqrt(self.beta1 );  # Normalize y to get v1 later.

    
        self.r2 = matrix(0.,(self.n,1))
        self.w = matrix(0.,(self.n,1))
        """
        %------------------------------------------------------------------
        % Initialize other quantities.
        % ------------------------------------------------------------------
        """
        self.oldb = 0; self.beta = copy(self.beta1); self.dbar = 0; self.epsln  = 0;
        self.qrnorm=copy(self.beta1); self.phibar=copy(self.beta1);self.rhs1=copy(self.beta1);
        self.rhs2= 0; self.tnorm2 = 0; self.ynorm2 = 0;
        self.cs     = -1;   self.sn     = 0;
        self.Arnorm = 0;
    
        scal(0., self.w)                         # w      = zeros(n,1);
        self.w2     = matrix(0., (n,1))
        self.r2 = copy(self.r1)                        # r2     = r1
        self.v =matrix(0., (n, 1))
        self.w1=matrix(0., (n,1))

        if self.option['show']: 
            print ' '
            print ' '
            head1 = '   Itn     x[0]     Compatible    LS';
            head2 = '         norm(A)  cond(A)';
            head2 +=' gbar/|A|';  # %%%%%% Check gbar
            print head1 + head2 

        #--------
    def iterate(self,times=1):
        'Apply specified number of minres iteratiosn'

        # Access variables held in the class
        shift = self.option['shift'];
        precon = self.precon; 
        eps=self.option['eps'];
        rtol = self.option['rtol']; 
        itnlim = self.option['itnlim']; 

        for i in range(times):
            #---------

            self.iter    = self.iter+1;
            """
            %-----------------------------------------------------------------
            % Obtain quantities for the next Lanczos vector vk+1, k = 1, 2,...
            % The general iteration is similar to the case k = 1 with v0 = 0:
            %
            %   p1      = Operator * v1  -  beta1 * v0,
            %   alpha1  = v1'p1,
            %   q2      = p2  -  alpha1 * v1,
            %   beta2^2 = q2'q2,
            %   v2      = (1/beta2) q2.
            %
            % Again, y = betak P vk,  where  P = C**(-1).
            % .... more description needed.
            %-----------------------------------------------------------------
            """
            s = 1/self.beta;                 # Normalize previous vector (in y).
            """
            v = s*y;                    # v = vk if P = I
            y = minresxxxA( A,v ) - shift*v;
            if itn >= 2, y = y - (self.beta/oldb)*r1; end
            """
            self.v = copy(self.y)
            scal(s, self.v)
            # Original function form, depreciated
            # A(v,y)
            gemv(self.A,self.v,self.y,'N')
            if abs(shift)>0:            axpy(self.v,self.y,-shift)
            if self.iter >= 2:          axpy(self.r1,self.y,-self.beta/self.oldb)

            alfa   = dotu(self.v, self.y)         # alphak
            axpy(self.r2, self.y, -alfa/self.beta)     # y    = (- alfa/self.beta)*r2 + y;
            
            # r1     = r2;
            # r2     = y;
            self.r1 = copy(self.y)
            _y=self.r1
            self.r1=self.r2
            self.r2=self.y
            self.y=_y

            if precon:  M(y)        # y = minresxxxM( M,r2 ); # end if
            self.oldb   = self.beta;              # oldb = self.betak
            self.beta   = dotu(self.r2,self.y)         # self.beta = self.betak+1^2
            if self.beta < 0: self.istop = 6;  break # end if
            self.beta   = sqrt(self.beta)
            self.tnorm2 = self.tnorm2 + alfa**2 + self.oldb**2 + self.beta**2


            if self.iter==1:                  # Initialize a few things.
                if self.beta/self.beta1 < 10*eps: # self.beta2 = 0 or ~ 0.
                    self.istop = -1;         # Terminate later.
                # end if
                # %self.tnorm2 = alfa**2  ??
                self.gmax   = abs(alfa)      # alpha1
                self.gmin   = copy(self.gmax)           # alpha1
            # end if
            """
            % Apply previous rotation Qk-1 to get
            %   [deltak self.epslnk+1] = [cs  sn][dbark    0   ]
            %   [gbar k dbar k+1]   [sn -cs][alfak self.betak+1].
            """
            oldeps = self.epsln
            delta  = self.cs*self.dbar + self.sn*alfa  # delta1 = 0         deltak
            gbar   = self.sn*self.dbar - self.cs*alfa  # gbar 1 = alfa1     gbar k
            self.epsln  =           self.sn*self.beta  # self.epsln2 = 0         self.epslnk+1
            self.dbar   =         - self.cs*self.beta  # self.dbar 2 = self.beta2     self.dbar k+1
            root   = sqrt(gbar**2 + self.dbar**2)
            self.Arnorm = self.phibar*root;       # ||Ar{k-1}||
            """
            % Compute the next plane rotation Qk
            gamma  = norm([gbar self.beta]); % gammak
            gamma  = max([gamma eps]);
            cs     = gbar/gamma;        % ck
            sn     = self.beta/gamma;        % sk
            """
            self.cs,self.sn,gamma=SymOrtho(gbar,self.beta)
            phi    = self.cs * self.phibar ;      # phik
            self.phibar = self.sn * self.phibar ;      # self.phibark+1

            """
            % Update  x.
            """
            denom = 1/gamma;
            """
            w1    = w2;
            w2    = w;
            w     = (v - oldeps*w1 - delta*w2)*denom;
            x     = x + phi*w;
            """
            self.w1 = copy(self.w)
            _w=self.w1
            self.w1=self.w2
            self.w2=self.w
            self.w=_w
            
            scal(-delta,self.w)
            axpy(self.w1,    self.w,-oldeps)
            axpy(self.v,     self.w)
            scal(denom, self.w)
            axpy(self.w,     self.x, phi)
            """
            % Go round again.
            """
            self.gmax   = max(self.gmax, gamma);
            self.gmin   = min(self.gmin, gamma);
            z      = self.rhs1/gamma;
            # ynorm2 = z**2  + ynorm2;
            ynorm2 = nrm2(self.x)**2
            #rhs1   = rhs2 - delta*z;
            #rhs2   =      - self.epsln*z;
            """
            % Estimate various norms.
            """
            Anorm  = sqrt( self.tnorm2 )
            ynorm  = sqrt( ynorm2 )
            epsa   = Anorm*eps;
            epsx   = Anorm*ynorm*eps;
            epsr   = Anorm*ynorm*rtol;
            diag   = gbar;
            if diag==0: diag = epsa;    # end if

            qrnorm = self.phibar;
            self.rnorm  = qrnorm;
            test1  = self.rnorm/(Anorm*ynorm); #  ||r|| / (||A|| ||x||)
            test2  = root / Anorm; # ||Ar{k-1}|| / (||A|| ||r_{k-1}||)
            """
            % Estimate  cond(A).
            % In this version we look at the diagonals of  R  in the
            % factorization of the lower Hessenberg matrix,  Q * H = R,
            % where H is the tridiagonal matrix from Lanczos with one
            % extra row, self.beta(k+1) e_k^T.
            """
            Acond  = self.gmax/self.gmin;
            """
            % See if any of the stopping criteria are satisfied.
            % In rare cases, self.istop is already -1 from above (Abar = const*I).
            """
            if self.istop==0:
                t1 = 1 + test1;       # These tests work if rtol < eps
                t2 = 1 + test2;
                if t2    <= 1      :self.istop = 2; # end if 
                if t1    <= 1      :self.istop = 1; # end if
                if self.iter   >= itnlim :self.istop = 5; # end if
                if Acond >= 0.1/eps:self.istop = 4; # end if
                if epsx  >= self.beta1  :self.istop = 3; # end if
                if test2 <= rtol   :self.istop = 2; # end if 
                if test1 <= rtol   :self.istop = 1; # end if

            """
            % See if it is time to print something.
            """

            prnt   = False;
            if self.n      <= 40       : prnt = True; # end if
            if self.iter    <= 10       : prnt = True; # end if
            if self.iter    >= itnlim-10: prnt = True; # end if
            if self.iter%10 == 0        : prnt = True  # end if
            if qrnorm <= 10*epsx  : prnt = True; # end if
            if qrnorm <= 10*epsr  : prnt = True; # end if
            if Acond  <= 1e-2/eps : prnt = True; # end if
            if self.istop  !=  0       : prnt = True; # end if

            if self.option['show'] and prnt:
                stself.r1 = '%6g %12.5e %10.3e' % ( itn, x[0], test1 );
                str2 = ' %10.3e'           % ( test2 );
                str3 = ' %8.1e %8.1e'      % ( Anorm, Acond );
                str3 +=' %8.1e'            % ( gbar/Anorm);
                print sts1, str2, str3
            # end if
            #if abs(self.istop) > 0: break;        # end if
            print 'end', sum(abs(self.r2))
            print self.rnorm

            #---------
        return self.x

    def solve(self):
        'Solve the linear equation sys by MINRES'
        while self.rnorm > self.option['rtol']:
            self.iterate()

        """
        % Display final status.
        """
        if self.option['show']:
            print " "
            print ' istop   =  %3g               itn   =%5g'% (istop,itn)
            print ' Anorm   =  %12.4e      Acond =  %12.4e' % (Anorm,Acond)
            print ' rnorm   =  %12.4e      ynorm =  %12.4e' % (rnorm,ynorm)
            print ' Arnorm  =  %12.4e' % Arnorm
            print msg[istop+2]
        return self.x, self.istop, self.iter, self.rnorm, self.Arnorm, self.Anorm, self.Acond, self.ynorm


