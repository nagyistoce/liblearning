/*
 * conjugate_gradient_optimizer.cpp
 *
 *  Created on: 2010-6-20
 *      Author: sun
 */

#include <liblearning/optimization/conjugate_gradient_optimizer.h>


#include <iostream>

conjugate_gradient_optimizer::conjugate_gradient_optimizer(int max_iter_, double ftol_):max_iter(max_iter_),ftol(ftol_)
{

}

conjugate_gradient_optimizer::~conjugate_gradient_optimizer()
{
}

#include <cmath>

#include <limits>

using namespace std;

template <typename F>
void mnbrak(double &ax, double &bx, double &cx, double &fa, double &fb, double &fc, 	const F & func);

template <typename F>
double brent(const double ax, const double bx, const double cx, const F & f,	const double tol, double &xmin);

void linmin(VectorXd &p, VectorXd &xi, double &fret, optimize_objective& obj);


tuple<double, VectorXd> conjugate_gradient_optimizer::optimize(optimize_objective& obj, const VectorXd & x0)
{

	// void NR::frprmn(Vec_IO_DP &p, const DP ftol, int &iter, DP &fret, DP func(Vec_I_DP &), void dfunc(Vec_I_DP &, Vec_O_DP &))

	const double EPS=1.0e-18;

	double gg,gam,fp,dgg;
	double fret;

	int n=x0.size();
	VectorXd p(n), g(n),h(n),xi(n);

	p = x0;
	tie(fp,xi) = obj.value_diff(p);

	g = - xi;
	xi = h = g;

	for (int its=0;its<max_iter;its++) {
		iter=its;

		linmin(p,xi,fret,obj);
		if (2.0*fabs(fret-fp) <= ftol*(fabs(fret)+fabs(fp)+EPS))
		{
			break;
		}
		tie(fp,xi) = obj.value_diff(p);
		gg = g.squaredNorm();
		dgg = xi.dot(xi+g);

		if (gg == 0.0)
			break;

		gam=dgg/gg;

		g = -xi;
		xi = h = g+gam*h;

	}

	return make_tuple(fret,p);

}




void linmin(VectorXd &p, VectorXd &xi, double &fret, optimize_objective& obj)
{
	int j;
	const double TOL=1.0e-8;
	double xx,xmin,fx,fb,fa,bx,ax;

	ax=0.0;
	xx=1.0;

	auto f = [&](double x)-> double {VectorXd xt = p + x*xi; return obj.value(xt);};
	mnbrak(ax,xx,bx,fa,fx,fb,f);
	fret=brent(ax,xx,bx,f,TOL,xmin);
	xi *= xmin;
	p += xi;

}

template<class T>
inline T SIGN(const T &a, const T &b)
	{return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);}


template <typename T>
inline void shft3(T &a, T &b, T &c, const T& d)
{
	a=b;
	b=c;
	c=d;
}

template <typename F>
void mnbrak(double &ax, double &bx, double &cx, double &fa, double &fb, double &fc, 	const F & func)
{
	const double GOLD=1.618034,GLIMIT=100.0,TINY=1.0e-20;
	double ulim,u,r,q,fu;

	fa=func(ax);
	fb=func(bx);
	if (fb > fa) {
		swap(ax,bx);
		swap(fb,fa);
	}
	cx=bx+GOLD*(bx-ax);
	fc=func(cx);
	while (fb > fc) {
		r=(bx-ax)*(fb-fc);
		q=(bx-cx)*(fb-fa);
		u=bx-((bx-cx)*q-(bx-ax)*r)/(2.0*SIGN(max(fabs(q-r),TINY),q-r));
		ulim=bx+GLIMIT*(cx-bx);
		if ((bx-u)*(u-cx) > 0.0) {
			fu=func(u);
			if (fu < fc) {
				ax=bx;
				bx=u;
				fa=fb;
				fb=fu;
				return;
			} else if (fu > fb) {
				cx=u;
				fc=fu;
				return;
			}
			u=cx+GOLD*(cx-bx);
			fu=func(u);
		} else if ((cx-u)*(u-ulim) > 0.0) {
			fu=func(u);
			if (fu < fc) {
				shft3(bx,cx,u,cx+GOLD*(cx-bx));
				shft3(fb,fc,fu,func(u));
			}
		} else if ((u-ulim)*(ulim-cx) >= 0.0) {
			u=ulim;
			fu=func(u);
		} else {
			u=cx+GOLD*(cx-bx);
			fu=func(u);
		}
		shft3(ax,bx,cx,u);
		shft3(fa,fb,fc,fu);
	}
}


template <typename F>
double brent(const double ax, const double bx, const double cx, const F & f,	const double tol, double &xmin)
{
	const int ITMAX=100;
	const double CGOLD=0.3819660;
	const double ZEPS=numeric_limits<double>::epsilon()*1.0e-3;
	int iter;
	double a,b,d=0.0,etemp,fu,fv,fw,fx;
	double p,q,r,tol1,tol2,u,v,w,x,xm;
	double e=0.0;

	a=(ax < cx ? ax : cx);
	b=(ax > cx ? ax : cx);
	x=w=v=bx;
	fw=fv=fx=f(x);
	for (iter=0;iter<ITMAX;iter++) {
		xm=0.5*(a+b);
		tol2=2.0*(tol1=tol*fabs(x)+ZEPS);
		if (abs(x-xm) <= (tol2-0.5*(b-a))) {
			xmin=x;
			return fx;
		}
		if (abs(e) > tol1) {
			r=(x-w)*(fx-fv);
			q=(x-v)*(fx-fw);
			p=(x-v)*q-(x-w)*r;
			q=2.0*(q-r);
			if (q > 0.0) p = -p;
			q=fabs(q);
			etemp=e;
			e=d;
			if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
				d=CGOLD*(e=(x >= xm ? a-x : b-x));
			else {
				d=p/q;
				u=x+d;
				if (u-a < tol2 || b-u < tol2)
					d=SIGN(tol1,xm-x);
			}
		} else {
			d=CGOLD*(e=(x >= xm ? a-x : b-x));
		}
		u=(fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
		fu=f(u);
		if (fu <= fx) {
			if (u >= x) a=x; else b=x;
			shft3(v,w,x,u);
			shft3(fv,fw,fx,fu);
		} else {
			if (u < x) a=u; else b=u;
			if (fu <= fw || w == x) {
				v=w;
				w=u;
				fv=fw;
				fw=fu;
			} else if (fu <= fv || v == x || v == w) {
				v=u;
				fv=fu;
			}
		}
	}
	xmin=x;
	return fx;
}
