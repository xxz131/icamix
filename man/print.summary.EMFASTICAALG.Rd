\name{print.summary.EMFASTICAALG}
\alias{print.summary.EMFASTICAALG}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
print.summary.EMFASTICAALG
}
\description{
\code{\link[base]{print}} method for class \code{summary.EMFASTICAALG}.
}
\usage{
\method{print}{summary.EMFASTICAALG}(x, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
  An \code{summary.EMFASTICA} object.
}
  \item{\dots}{
%%     ~~Describe \code{\dots} here~~
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
Returned (invisibly) is the full value of \code{x} itself.
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
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
print(summary(b))
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
