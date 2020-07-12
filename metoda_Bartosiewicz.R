n = ncol(x)
t_alpha = qt(0.995, df=n-2) 
r_star = sqrt(t_alpha^2 / (t_alpha^2 + n - 2))


R0 = replicate(n, 0)

for (i in 1:n )
{
  R0[i] = cor(x[,i],data.train$Y, method = "pearson")
}

index = c()

for (i in 1:n )
{
  if (abs(R0[i]) <= r_star)
  {
    index = c(index,i)
    
  }
}

R0 = R0[-length(R0)]
absmax <- function(x) { x[which.max( abs(x) )]}
rh = which(R0==absmax(R0))


Rhi = replicate(n, 0)

for (i in 1:n )
{
  Rhi[i] = cor(x[,i],x[,rh], method = "pearson")
}


for (i in 1:n )
{
  if (abs(Rhi[i]) > r_star & !is.element(i, index))
  {
    index = c(index,i)
    
  }
}

data.train = subset(data.train, select = -index)

dim(data.train)