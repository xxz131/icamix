\name{plot.EMFASTICAALG}
\alias{plot.EMFASTICAALG}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
plot.EMFASTICAALG
}
\description{
\code{plot} method for class \code{EMFASTICAALG}.
}
\usage{
\method{plot}{EMFASTICAALG}(x, vec1 = c(1:2), vec2 = c(1:2), ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
  An \code{EMFASTICAALG} object.
}
  \item{vec1}{
An integer vector of length two specifying the coordinates with respect to which the data is scatter plotted.
}
  \item{vec2}{
  An integer vector of length two specifying the coordinates with respect to which the original signal for each mixture component is scatter plotted.
}
  \item{\dots}{
%%     ~~Describe \code{\dots} here~~
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{Returned (invisibly) is the full value of \code{x} itself.
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
plot(b)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
