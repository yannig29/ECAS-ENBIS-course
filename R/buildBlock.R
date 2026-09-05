
buildBlock <- function (Nblock, data0) 
{
  borne_block <- seq(1, nrow(data0), length = Nblock + 1) %>% 
    floor
  block_list <- list()
  l <- length(borne_block)
  for (i in c(2:(l - 1))) {
    block_list[[i - 1]] <- c(borne_block[i - 1]:(borne_block[i] - 
                                                   1))
  }
  block_list[[l - 1]] <- c(borne_block[l - 1]:(borne_block[l]))
  return(block_list)
}


buildBlock_random <- function (Nblock, data0) 
{
  n <- nrow(data0)
  size <- floor(n/Nblock)
  block_list <- list()
  for (i in 1:Nblock) {
    block_list[[i]] <- sample(c(1:n), size, replace = FALSE)
  }
  return(block_list)
}


