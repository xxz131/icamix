// EMInterwovenFastICA.cpp - Implementation of EM-Like algorithm interwoven with weighted symmetric FastICA
// Written by Xiaotian Zhu

#include <RcppArmadillo.h>
#include <algorithm>
#include <vector>
#include <cmath>

#include <iostream>
#include <string>
#include <stdio.h>
#include <time.h>




using namespace Rcpp;
using namespace arma;

#define InvSqr2pi             0.39894228040143267794



List WtsFastICA(const arma::mat& X, const arma::vec& wts, const arma::mat& Wconst, bool verbose = true, double alpha = 1, int maxIteration = 60, double tolerance = 1e-6);

// [[Rcpp::export]]
List EMInterwovenFastICA(
	const arma::mat& DataMtr,
	const arma::mat& MembershipProbs_,
	double bandWidth,
	int maxIteration = 300,
	int icaIteration = 150,
	double tolerance = 1e-6,
	bool verbose = true,
	bool combine = true
	)
{
	// 1. Read in the data and other arguments

	int r = DataMtr.n_rows; // data dimension
	int n = DataMtr.n_cols; // data size


	// 2. Set initial values and prepare

	int index = 0; // iteration counting
	double difference = 1000; // "distance" between the current estimate with the previous one

	mat MembershipProbs = MembershipProbs_; // n by nCluster
	int nCluster = MembershipProbs.n_cols;

	mat Lambdas(maxIteration, nCluster, fill::zeros); // a maxIteration by nCluster matrix of which the rows are holding estimates of lambda's: the mixing probabilities
	mat ObjValue(maxIteration, 1, fill::zeros); // value for the optimization objective function

	mat ICABandWidth(nCluster, 1, fill::zeros); // values for ICA bandwidth for each independent signal of each mixture component

	//std::vector<mat> Densities(r); // r nCluster by n matrices for holding marginal density estimates (for the original independent signals!!!)
	//for(int i = 0; i < r; i++)
	//{
	//	Densities[i] = mat(nCluster, n, fill::zeros);
	//}

	mat ProductDensity(nCluster, n, fill::ones); // a nCluster by n matrices for holding product densities

	std::vector<mat> WMtrs(nCluster); // nCluster r by r matrices for holding the unmixing matrix for each cluster
	for(int i = 0; i < nCluster; i++)
	{
		WMtrs[i] = eye<mat>(r,r);
	}

	std::vector<mat> WUnmixZ(nCluster); // nCluster r by r matrices for holding estimated UnmixZ matrix for each cluster; first set to 1 by 1 which will be dealt with by WtsFastICA
	if(combine == true)
	{
		for(int i = 0; i < nCluster; i++)
		{
			WUnmixZ[i] = eye<mat>(1,1);
		}
	} else {
		for(int i = 0; i < nCluster; i++)
		{
			WUnmixZ[i] = eye<mat>(r,r);
		}
	}

	std::vector<mat> OriginalSignals(nCluster); // nCluster r by n matrices for holding original signals for each cluster
	for(int i = 0; i < nCluster; i++)
	{
		OriginalSignals[i] = DataMtr;
	}


	// 3. Enter Iterations

	Lambdas.row(0) = mean(MembershipProbs, 0);

	if(verbose == true)
	{
		Rcpp::Rcout << "iter " << 0 << "    obj " << ObjValue(0,0) << "    diff " << difference << "    lambda" << Lambdas.row(0) << endl;
	}

	while (difference > tolerance && index < (maxIteration-1))
	{

		// 2) ICA step

		ProductDensity = mat(nCluster, n, fill::ones);

		if(combine == true)
		{

			for(int i = 0; i < nCluster; i++)
			{
				List TempICAResult = WtsFastICA(DataMtr, MembershipProbs.col(i), WUnmixZ[i], true, 1, icaIteration);

				WMtrs[i] = mat(as<NumericMatrix>((TempICAResult)["UnmixX"]).begin(), r, r);

				WUnmixZ[i] = mat(as<NumericMatrix>((TempICAResult)["UnmixZ"]).begin(), r, r);

				OriginalSignals[i] = mat(as<NumericMatrix>((TempICAResult)["IndSignal"]).begin(), r, n, false, false);	
			}

		}


		// 3) Density estimation step

		//KDEest(OriginalSignals, MembershipProbs, bandWidth, ProductDensity, ICABandWidth);
		{
			// calculates kernel density estimates
			int nCluster = ProductDensity.n_rows;
			double tempsum, temph, temp;
			rowvec tempVec(n, fill::zeros);

			for(int i = 0; i < nCluster; i++)
			{
				tempsum = sum(MembershipProbs.col(i));
				if(bandWidth == 0){
					temph = 0.5 / std::pow(tempsum, 0.2);
				} else {
					temph = bandWidth;
				}

				ICABandWidth(i,0) = temph;

				for(int j = 0; j < r; j++)
				{
					tempVec.zeros();
					for(int pj = 0; pj < n; pj++)
					{
						for(int dj = pj; dj < n; dj ++)
						{
							temp = 1/temph * InvSqr2pi * std::exp(-0.5*std::pow((OriginalSignals[i](j,pj)-OriginalSignals[i](j,dj))/temph, 2));
							tempVec(pj) += temp * MembershipProbs(dj,i);
							if(dj != pj)
							{
								tempVec(dj) += temp * MembershipProbs(pj,i);
							}
						}
						ProductDensity(i,pj) *= tempVec(pj)/tempsum;
					}

				}

			}

		}

		for (int i = 0; i < nCluster; i++)
		{
			ProductDensity.row(i) = (Lambdas(index, i) * std::abs(det(WMtrs[i]))) * ProductDensity.row(i);
		}


		// 4) E-step

		for(int i = 0; i < n; i++)
		{
			double tempDenom = sum(ProductDensity.col(i));

			for(int j = 0; j < nCluster; j++)
			{
				MembershipProbs(i, j) = ProductDensity(j, i)/tempDenom;
			}

			ObjValue(index+1,0) += log(tempDenom);
		}


		// 1) M-step

		Lambdas.row(index+1) = mean(MembershipProbs, 0);


		// Preparing for next iteration

		difference = max( abs(Lambdas.row(index+1)-Lambdas.row(index)) );


		// print out results of this iteration if verbose = TRUE
		if(verbose == true)
		{
			Rcpp::Rcout << "iter " << index+1 << "    obj " << ObjValue(index+1,0) << "    diff " << difference << "    lambda" << Lambdas.row(index+1) << endl;
		}

		index += 1;
	}


	// 4. Post-processing

	Lambdas = Lambdas(span(0, index), span::all);
	ObjValue = ObjValue(span(0, index), 0);

	return List::create(
		Named("InputData") = DataMtr,
		Named("Lambdas") = Lambdas,
		Named("WMtrs") = WMtrs,
		Named("WUnmixZ") = WUnmixZ,
		Named("OriginalSignals") = OriginalSignals,
		Named("ProductDensity") = ProductDensity,
		Named("MembershipProbs") = MembershipProbs,
		Named("ObjValue") = ObjValue,
		Named("ICABandWidth") = ICABandWidth
		); 
}
