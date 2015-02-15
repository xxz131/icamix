\name{CLASSDIFFRATE}
\alias{CLASSDIFFRATE}
\title{
CLASSDIFFRATE
}
\description{
A function calculates classification difference rate between two factors. It is used in interpreting info stored in EMFASTICA object.
}
\usage{CLASSDIFFRATE(factor1, factor2)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{factor1}{
First factor.
}
  \item{factor2}{
Second factor of the same length as the First factor.
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
\item{answer }{The percentage of instances when factor1[i] is not equal factor2[i].}
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
\examples{## An example evaluates the classification difference rate
## between two classification results in the form of factors
fac1<-factor(c(1,4,2,3,1,1,3,3,1,2,2,1))
fac2<-factor(c(3,1,2,2,1,2,4,3,2,3,1,1))
CLASSDIFFRATE(fac1, fac2)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
