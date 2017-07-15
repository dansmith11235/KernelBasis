#ilibrary's needed
library(cluster)
library(HSAUR)
library(ggfortify)
library(ggplot2)
library(dbscan)


# set margin sizes
par(mar=c(5,5,5,5))

# allows the user to choose the shakespeare file from their directory
myFile <- file.choose()

#reads the file into an R dataframe
shakespeare <- read.csv(myFile)

#creates a list of names of the plays
names <- shakespeare[,1]

# names each row the name of the play and act
row.names(shakespeare) <- names

#remove column that had all the names
# for clustering algorithms data frame needs to numerical
df <- shakespeare[c(2:6)]

#k means results for two clusters
kmres2 <-kmeans(df,2,iter.max = 1000000)

#silhoutte plot for the kmres2 
dissE <- daisy(df)
dE2   <- dissE^2
sk2   <- silhouette(kmres2$cl, dE2)
plot(sk2)


# used to make an elbow plot for determing size of k
wss <- (nrow(df)-1)*sum(apply(df,2,var))
for (i in 2:10) wss[i] <- sum(kmeans(df,
                                     centers=i)$withinss)
plot(1:10, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares",)

title(main = "Elbow Method for Choosing K")

#k means results for four clusters
kmres4 <-kmeans(df,4,iter.max = 1000000)

#silhoutte plot for the kmres4
dissE <- daisy(df)
dE2   <- dissE^2
sk2   <- silhouette(kmres4$cl, dE2)
plot(sk2,main = "Shakespeare Clustering Silhouette Plot with  K = 4")


#use principal component analysis
df.pca <- prcomp(df, center = TRUE, scale. = TRUE)

#creates PCA graph for kres2 with labels  
set.seed(1)
autoplot(kmres2,data = df, label = TRUE, label.size = 3)

#creates PCA graph for kres2 with no labels  
set.seed(1)
autoplot(kmres2,data = df)

#generates the OPTICS plot for choosing
#DBscan parameters
opt <- optics(as.matrix(df),2)
plot(opt)

#db scan only generates one cluster
db <- dbscan(df,.08,5)
# ones under zero weren't classified to a cluster
db

