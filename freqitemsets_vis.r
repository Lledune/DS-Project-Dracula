rm(list=ls())

setwd("D:/Google Drive/DATS2MS/LINMA2472 Algorithms in data science/Project/Dracula")
data <- read.csv("freqitemsets.csv", sep = ";", header = F)
data[,1] <- as.factor(data[,1])

png("Images/fisplot.png", 1820, 1080, res = 200)
bp <- barplot(data[data$V1 == 23,]$V3, ylab = "Character sets", xlab = "Together in sentences",
              border = NA, col = "#0049be", horiz = T, main = "Frequency of character sets in chapter XXIII")
inside = (data[data$V1 == 23,]$V3 <= 50)
text(x = data[data$V1 == 23,]$V3[inside], y = bp[inside], labels = data[data$V1 == 23,]$V2[inside], pos = 4, col = "#0049be")
text(x = data[data$V1 == 23,]$V3[!inside], y = bp[!inside], labels = data[data$V1 == 23,]$V2[!inside], pos = 2, col = "white")
dev.off()