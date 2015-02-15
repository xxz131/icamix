// WtsKde.cpp - Implementation of weighted kernel density estimation for one dimension
// Written by Xiaotian Zhu

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

#define InvSqr2pi             0.39894228040143267794

// [[Rcpp::export]]
List WtsKde(
	const arma::vec& X,
	const arma::vec& wts,
	const arma::vec& grid,
	double h
	) 
{
	int n1 = X.size(); // number of data points
	int n2 = grid.size(); // number of grid points where we need to evaluate the kernal density estimate

	arma::vec answer(n2, fill::zeros); // the answer to be returned
	
	for(int i = 0; i < n2; i++){
		for(int j = 0; j < n1; j++){
			answer(i) += 1/h * InvSqr2pi * std::exp(-0.5*std::pow((grid(i)-X(j))/h, 2)) * wts(j);
		}
	}

	return List::create(
		Named("weightedkde") = answer
		); 
}