// WeightedFastICA.cpp - Implementation of weighted symmetric FastICA
// Written by Xiaotian Zhu

#include <RcppArmadillo.h>
#include <algorithm>

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List WtsFastICA(
	const arma::mat& X,
	const arma::vec& wts,
	const arma::mat& Wconst,
	bool verbose = true,
	double alpha = 1,
	int maxIteration = 60,
	double tolerance = 1e-6
	) 
{
	/* Weigted symmetric FastICA with Initial W input*/

	// 1. Reading in the arguments and preprocessing

	unsigned int r = X.n_rows; // data dimension
	unsigned int n = X.n_cols; // data size

	if(n != wts.size())
		stop("Length of input weights vector doesn't match size of the data.");

	if(alpha < 1 || alpha > 2)
		stop("Value of alpha should be in [1,2]");


	// 2. Centering

	mat meanX = X * mat(wts) / sum(wts);
	mat centeredX(r, n);
	for(unsigned int i = 0; i < r; i++)
	{
		for(unsigned int j = 0; j < n; j++)
		{
			centeredX(i, j) = X(i, j) - meanX(i, 0);
		}
	}


	// 3. Decorrelating

	mat CovX = centeredX * diagmat(wts)* centeredX.t() / sum(wts);

	mat U_CovX;
	vec s_CovX;
	mat V_CovX;
	svd_econ(U_CovX, s_CovX, V_CovX, CovX, "both", "dc"); // economic singular decomosition

	mat DecorrelatingMatrix = solve(diagmat(sqrt(s_CovX)), U_CovX.t());

	mat Z = DecorrelatingMatrix * centeredX;


	// 4. Symmetric FastICA using logcosh approximation to neg-entropy

	mat U_W;
	vec s_W;
	mat V_W;

	mat Wprevious;
	mat W;

	if(r != Wconst.n_rows || r != Wconst.n_cols)
	{
		GetRNGstate();
		W = mat(r, r, fill::zeros); // generate W's initial value
		for(unsigned int i = 0; i < r; i++)
		{
			for(unsigned int j = 0; j < r; j++)
			{
				W(i, j) = unif_rand()+0.1*i*i;
			}
		}
		PutRNGstate();
		svd_econ(U_W, s_W, V_W, W, "both", "dc");
		W = U_W * V_W.t();
		//W = eye<mat>(r, r);
		Wprevious = W;
	}
	else
	{
		W = Wconst; // get W's initial value from input		
		Wprevious = W;
	}

	mat GWZ;
	mat GprimeWZ;
	double difference = 1000;
	int index = 1;

	while(difference > tolerance && index <= maxIteration)
	{
		GWZ = tanh(alpha*W*Z);
		GprimeWZ = alpha * (1 - square(GWZ));
		W = GWZ*diagmat(wts)*Z.t()/sum(wts) - diagmat(GprimeWZ*mat(wts)/sum(wts)) * W;
		svd_econ(U_W, s_W, V_W, W, "both", "dc");
		W = U_W * V_W.t();
		difference = max(abs(abs(diagvec(W * Wprevious.t())) - 1));
		Wprevious = W;
		if (verbose)
			Rcpp::Rcout << "          ICAiter " << index << "  diffICA " << difference << endl;
		index += 1;
	}


	// 5. Postprocessing

	mat UnmixX = W * DecorrelatingMatrix;
	mat IndSignal = UnmixX * centeredX;
	//mat MixMtr = inv(UnmixX);

	return List::create(
		//Named("InputData") = X,
		//Named("CenteredData") = centeredX,
		//Named("DecorMtr") = DecorrelatingMatrix,
		//Named("WhitenedData") = Z,		
		Named("UnmixZ") = W,
		Named("UnmixX") = UnmixX,
		Named("IndSignal") = IndSignal//,
		//Named("MixMtr") = MixMtr
		); 

	//return UnmixX;

}
