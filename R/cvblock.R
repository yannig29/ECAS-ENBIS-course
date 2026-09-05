

forecastBlock<-function(block, formula, data)
{
  g<- gam(formula, data=data[-block,] )
  forecast<-predict(g, newdata=data[block,])
  return(forecast)
}


forecastBlock2<-function(block, formula, data)
{
  g<- bam(formula, data=data[-block,], discrete=T)
  forecast<-predict(g, newdata=data[block,])
  return(forecast)
}




# blockBuild <- function(Nblock, data)
# {
#   borne_block <- seq(1, nrow(data), length=Nblock+1)
#   borne_block <- floor(borne_block)
#   block_list<-list()
#   l<-length(borne_block)
#   for(i in c(2:(l-1)))
#   {
#     block_list[[i-1]] <- c(borne_block[i-1]:(borne_block[i]-1))
#   }
#   block_list[[l-1]]<-c(borne_block[l-1]:(borne_block[l]))
#   return(block_list)
# }
