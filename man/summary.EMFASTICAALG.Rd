\name{summary.EMFASTICAALG}
\alias{summary.EMFASTICAALG}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
summary.EMFASTICAALG
}
\description{
\code{\link[base]{summary}} method for class \code{EMFASTICAALG}.
}
\usage{
\method{summary}{EMFASTICAALG}(object, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{
An EMFASTICAALG object.
}
  \item{\dots}{
%%     ~~Describe \code{\dots} here~~
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{The returned value is a "summary.EMFASTICAALG" object which consists a list:
%%  ~Describe the value returned
%%  If it is a LIST, use
\item{$inputData }{A matrix of which the columns are data entries. Its dimension is \code{r} by \code{n}.}
\item{$originSig }{List of Recovered ICA components for each of the \code{m} clusters.}
\item{$call }{The function call which results in the corresponding EMFASTICAALG object.}
\item{$time }{Computing time elaped in second.}
\item{$numIter }{Total number of iterations.}
\item{$lastObj }{Objective function value from the last iteration.}
\item{$compMeans }{Means of each mixture component.}
\item{$compVars }{Covariances of each mixture component.}
\item{$numObs }{Total number of observations.}
\item{$numAtr }{Dimension of data points.}
\item{$numCls }{Number of mixture components.}
\item{$estWts }{Estimated mixing weights.}
\item{$estCls }{A factor whose levels represent estimated class membership.}
\item{$dataLklhd }{A vector storing data loglikelihood from each iteration.}
%% ...
}
%\references{
%% ~put references to the literature/web site here ~
%}
%\author{
%%  ~~who you are~~
%}
%\note{
%%  ~~further notes~~
%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%}
\examples{
## An Example that runs the NSMM-ICA algorithm on Cohen's tone data
library(mixtools)
data(tonedata)

b <- EMFASTICAALG(tonedata, 2, h=0, tol=1e-8)
summary(b)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
