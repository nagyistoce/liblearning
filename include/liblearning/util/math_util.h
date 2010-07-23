/*
 * math_util.h
 *
 *  Created on: 2010-6-4
 *      Author: sun
 */

#ifndef MATH_UTIL_H_
#define MATH_UTIL_H_


void init_math_utils();

void finish_math_utils();

#ifdef USE_MKL
void matrix_linear_transform(double * Y, int rW, int cW, int cX, const double alpha, const double * W, const char Tans, const double * X, double beta, const double * b );

void matrix_logistic_transform(double * Y, int rW, int cW, int cX, const double alpha, const double * W, const char Tans, const double * X, double beta, const double * b );

void matrix_dot(double * Y, int rW, int cW, int rX, int cX, const double alpha, const double * W, const char TransW, const double * X,const char TransX );

void vector_linear_transform(double * y, int rW, int cW, const double alpha, const double * W, const char Trans, const double * x);

void fill_randn(double * data, int n,  double mean, double sigma);

void fill_rand(double * data, int n, double l, double h);

#endif



#endif /* MATH_UTIL_H_ */
