#include <mex.h>
#include <math.h>
#include <matrix.h>

int sign(double x);
double sabs(double x);
double max0(double x);
double smin(double x, double y);

void mexFunction(const int nlhs, mxArray *plhs[],
                 const int nrhs, const mxArray *prhs[])
{
    int m, n, i;
    double *opt_val, *opt, *costmat, *alphamat, *lambda2;
    double ct1, ct2, ct3, ap1, ap2, g, lam, valminp, valap1, valap2, minp, sopt, sopt_val;
    double opt_tmp, optval_tmp;
    
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    opt_val = mxGetPr(plhs[0]);
    opt = mxGetPr(plhs[1]);

    costmat = mxGetPr(prhs[0]);
    alphamat = mxGetPr(prhs[1]);
    lambda2 = mxGetPr(prhs[2]);
    lam = lambda2[0];
    m = mxGetM(prhs[0]);
    n = mxGetN(prhs[0]);

    for (i=0; i<m; i++)
    {
       ct1 = costmat[i]; ct2 = costmat[i+m]; ct3 = costmat[i+2*m];
       ap1 = alphamat[i]; ap2 = alphamat[i+m];
       g = ct2/ct1*(-0.5);
       minp = g*sign(max0(sabs(g) - sqrt(lam/ct1)));
       valminp = ct1*minp * minp + ct2*minp + ct3 + lam * sabs(sign(minp)) * ct1 * 2;
       valap1 = ct1*ap1*ap1 + ct2*ap1 + ct3 + lam * sabs(sign(ap1)) * ct1 * 2;
       valap2 = ct1*ap2*ap2 + ct2*ap2 + ct3 + lam * sabs(sign(ap2)) * ct1 * 2;
       if (valminp <= smin(valap1, valap2) && minp <= ap2 && minp >= ap1)
       {sopt_val = valminp;
        sopt = minp;}
       else if (valap1 <= valap2)
       {sopt_val = valap1;
        sopt = ap1;}
       else
       {sopt_val = valap2;
        sopt = ap2;}
       if (i==0) {optval_tmp = sopt_val; opt_tmp = sopt;}
        else{ if (optval_tmp > sopt_val) {optval_tmp = sopt_val; opt_tmp = sopt; }}
    }
    opt_val[0] = optval_tmp;
    opt[0] = opt_tmp;
}

int sign(double x)
{
    if (x>0) {return 1;} else if (x==0) {return 0;} else {return -1;} 
}

double sabs(double x)
{
if (x<0) {return -x;} else {return x;}
}

double max0(double x)
{
    if (x>0) {return x;} else {return 0;}
}

double smin(double x, double y)
{
    if (x>=y) {return y;} else {return x;}
}
