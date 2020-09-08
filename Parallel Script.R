start_time = Sys.time()


library(parallel); library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv",
                           number = 3,
                           allowParallel = TRUE)
model = train(..., 
              trcontrol=fitControl) 

stopCluster(cluster)
registerDoSEQ()

end_time = Sys.time()
paste("Time Taken = ", end_time - start_time)