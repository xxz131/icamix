\name{EMFASTICAALG}
\alias{EMFASTICAALG}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
EMFASTICAALG
}
\description{
An R wrapper for carrying out NSMM-ICA on nonparametric multivariate ICA mixture data.
}
\usage{
EMFASTICAALG(DataMatrix, numCluster, h = 0, maxiter = 300, 
icaiter = 150, tol = 1e-06, verb = TRUE, combine = TRUE,
seednum = 82196, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{DataMatrix}{
  A matrix of which the rows are data entries. Its dimension is \code{n} by \code{r}.
}
  \item{numCluster}{
  Predetermined number of mixing components \code{m}.
}
  \item{h}{
  Bandwidth. If \code{h} is set equal zero (default), iterative bandwidth selection will be used.
}
  \item{maxiter}{
  Maximum number of iterations. Default is \code{300}.
}
  \item{icaiter}{
  Maximum number of ICA iterations in each step. Default is \code{150}.
}
  \item{tol}{
  Threshold that defines convergence (of the outer loop). Default is \code{1}\code{e}-\code{6}.
}
  \item{verb}{
  \code{TRUE} (default) or \code{FALSE} indicating whether to print out info at each iteration.
}
  \item{combine}{
  \code{TRUE} (default) or \code{FALSE} indicating whether to implement the ICA step.
}
  \item{seednum}{
  Seed number (default is \code{82196}) used in kmeans before 1st iteration.
}
  \item{\dots}{
%%     ~~Describe \code{\dots} here~~
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
The returned value is an \code{EMFASTICAALG} object which consists of a list of items:
\item{$InputData }{A matrix of which the columns are data entries. Its dimension is \code{r} by \code{n}.}
\item{$Lambdas }{A matrix where rows store estimated mixing weights from each iteration.}
\item{$WMtrs }{List of \code{r} by \code{r} unmixing matrices for each of the m clusters.}
\item{$WUnmixZ }{List of unmixing matrices for whitened data for each of the m clusters.}
\item{$OriginalSignals }{List of Recovered ICA components for each of the \code{m} clusters.}
\item{$ProductDensity }{\code{m} by \code{n} matrix where each row stores the estimated density value of the observed data points for each of the m clusters.}
\item{$MembershipProbs }{\code{n} by \code{m} matrix where each row stores the component membership probabilities of the corresponding data point.}
\item{$ObjValue }{Vector holding values of data loglikelihood.}
\item{$ICABandWidth }{Matrix holding choices of bandwidth for original signals.}
\item{$call }{The function call that results in the returned object.}
\item{$time }{Computing time elapsed in second.}
%%  \item{comp2 }{Description of 'comp2'}
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
  
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
