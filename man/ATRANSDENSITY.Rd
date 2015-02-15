\name{ATRANSDENSITY}
\alias{ATRANSDENSITY}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
ATRANSDENSITY
}
\description{
A function evaluates density of linearly transformed random vector on a given grid. It is used in processing EMFASTICAALG object to obtain density estimation of the mixture components.
}
\usage{
ATRANSDENSITY(grid, A, f)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{grid}{
  A matrix whose columns store the grid points.
}
  \item{A}{
  Matrix for the linear transformation.
}
  \item{f}{
  Density function before the linear transformation.
}
}
%\details{
%%  ~~ If necessary, more details than the description above ~~
%}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
\item{answer}{Matrix of the same size as grid, with each element being the evaluated linear transformed density at the corresponding grid point.}
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
## An example that evaluates the 2-D uniform distribution on a linear transformation of [1,3]x[1,3]
## f1ind is the density of the uniform distribution on [1,3]^r
f1ind <- function(grid){# mixture component 1 original signal density function
  n <- ncol(grid)
  r <- nrow(grid)
  answer <- rep(1,n)
  for(i in 1:n){
    for(j in 1:r){
      answer[i] <- answer[i] * (grid[j,i] >= 1 & grid[j,i] <= 3) / 2
    }
  }
  answer
}

A <- matrix(c(6, 9, -12, 15), 2, 2, byrow = FALSE)

gridpoints <- t(as.matrix(expand.grid(seq(-32,12,2),seq(18,80,2))))

f1trans <- ATRANSDENSITY(gridpoints, A, f1ind)

plot(t(gridpoints),col=(f1trans>0))
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
