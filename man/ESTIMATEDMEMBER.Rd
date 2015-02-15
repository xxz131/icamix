\name{ESTIMATEDMEMBER}
\alias{ESTIMATEDMEMBER}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
ESTIMATEDMEMBER
}
\description{
A function calculates estimated class membership from an EMFASTICAALG object.
}
\usage{
ESTIMATEDMEMBER(rstICAMIX)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{rstICAMIX}{
  An EMFASTICAALG object.
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
\item{estimatedmember }{A factor with levels representing the estimated classes.}
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
estimatedgroup <- ESTIMATEDMEMBER(b) # estimated species info

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
