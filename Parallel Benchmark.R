# Prerequisite: Selecting a Machine Learning Problem ----------------------
intervalStart <- Sys.time()
library(mlbench)
data(Sonar)
library(caret)
set.seed(95014)

# create training & testing data sets
inTraining <- createDataPartition(Sonar$Class, p = .75, list=FALSE)
training <- Sonar[inTraining,]
testing <- Sonar[-inTraining,]
# set up x and y to avoid slowness of caret() with model syntax
y <- training[,61]
x <- training[,-61]


# Step 1: Configure parallel processing -----------------------------------
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)


# Step 2: Configure trainControl object -----------------------------------
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)


# Step 3: Develop training model ------------------------------------------
system.time(fit <- train(x,y, method="rf",data=Sonar,trControl = fitControl))


# Step 4: De-register parallel processing cluster -------------------------
stopCluster(cluster)
registerDoSEQ()


# End --------------------------------------------------------------

# Results
# user  system elapsed 
# 0.50    0.00    3.39 

# Reference:
# https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md
