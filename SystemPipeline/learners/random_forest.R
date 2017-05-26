library(mlr)

# read data
df <- read.table('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/dataset.new.rawdata.noseg.dump.nobrackets.csv', header=TRUE, colClasses=c('character', 'complex'), sep=',')

# view data
View(df)

# build task for MLR
task <- makeClassifTask(id='alphabets', data=df, target='label')

# build resampler for MLR, use Cross-Validation
resampler <- makeResampleDesc(method = 'CV')

# build random forest classifier
random.forest.learner <- makeLearner(cl = 'classif.randomForest', id = 'Random Forest')

# apply classifier on given data
r.random.forest <- resample(learner=random.forest.learner, task=task, resampling=resampler, show.info=FALSE, models=TRUE)

# display results
r.random.forest
