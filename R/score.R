rmse<-function(eps)
{
  return(round(sqrt(mean(eps^2,na.rm=TRUE)),digits=0))
}


rmse2<-function(y, ychap)
{
  eps=y-ychap
  return(round(sqrt(mean(eps^2,na.rm=TRUE)),digits=0))
}


mape<-function(y,ychap)
{
  return(round(100*mean(abs(y-ychap)/abs(y)),digits=2))
}
